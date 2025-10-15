# Get project root directory
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
import gc

class IntentionEvaluator:
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.init_qwen25_model()
        self.results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "detailed_results": [],
            "task_type": "intention_analysis"
        }
        self.dialogue_summaries = {}
        self.context_patterns = {}

    def init_qwen25_model(self):
        """Initialize Qwen2.5 70B model and tokenizer"""
        model_path = os.path.join(project_root, "qwen_models", "Qwen_Qwen2.5-72B-Instruct")

        print(f"🚀 GPU {self.gpu_id}: 加载Qwen2.5-72B-Instruct模型...")
        print(f"🔧 GPU {self.gpu_id}: 使用设备: {self.device}")

        try:
            # 设置GPU设备并清理内存
            torch.cuda.set_device(self.gpu_id)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            print(f"📥 GPU {self.gpu_id}: 加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )

            print(f"📥 GPU {self.gpu_id}: 加载模型到GPU...")
            try:
                # 尝试使用flash attention
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",  # Auto-distribute across available GPUs for 72B model
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2"
                )
            except Exception:
                print(f"⚠️ GPU {self.gpu_id}: Flash Attention不可用，使用默认attention")
                # 回退到默认attention
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",  # Auto-distribute across available GPUs for 72B model
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )

            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated(self.gpu_id) / 1e9
                print(f"✅ GPU {self.gpu_id}: Qwen2.5-7B-Instruct模型加载完成! 内存使用: {memory_after:.2f}GB")

        except Exception as e:
            print(f"❌ GPU {self.gpu_id}: 模型加载失败: {e}")
            torch.cuda.empty_cache()
            raise e

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer"""
        try:
            tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            return tokens.input_ids.shape[1]
        except:
            # Fallback: rough estimation
            return len(text) // 4

    def reduce_conversations_to_fit_tokens(self, context: str, max_tokens: int, use_cumulative_context: bool = True) -> str:
        """Reduce number of conversations instead of hard truncation to fit token limit"""
        current_tokens = self.count_tokens(context)

        if current_tokens <= max_tokens:
            return context

        print(f"⚠️ GPU {self.gpu_id}: Context too long ({current_tokens} tokens), reducing conversations to fit {max_tokens} tokens")

        # If no previous conversations, fall back to smart truncation
        if "Previous Conversations" not in context or not use_cumulative_context:
            return self.truncate_context_smartly(context, max_tokens)

        # Split context into background and conversation parts
        parts = context.split("\\n\\nPrevious Conversations")
        base_part = parts[0]  # Background context
        base_tokens = self.count_tokens(base_part)

        # If base part is already too long, truncate it first
        if base_tokens > max_tokens - 500:  # Reserve 500 tokens for at least some conversation
            print(f"    ⚠️ Background context itself is too long ({base_tokens} tokens), truncating background first")
            base_part = self.truncate_context_smartly(base_part, max_tokens - 500)
            base_tokens = self.count_tokens(base_part)

        if len(parts) > 1:
            # We have previous conversations, reduce them gradually
            remaining_tokens = max_tokens - base_tokens - 200  # Reserve for formatting

            # Parse conversation section
            prev_conv_section = parts[1]
            conversations_text = prev_conv_section.split("\\n\\n---\\n\\n")

            # Start with most recent conversations and add until we hit limit
            conversations_to_keep = []
            current_conv_tokens = 0

            # conversations_text[0] contains the header, conversations_text[1:] contain actual conversations
            if len(conversations_text) > 1:
                for conv_text in reversed(conversations_text[1:]):  # Most recent first
                    conv_tokens = self.count_tokens(conv_text)
                    if current_conv_tokens + conv_tokens <= remaining_tokens:
                        conversations_to_keep.insert(0, conv_text)  # Insert at beginning for chronological order
                        current_conv_tokens += conv_tokens
                        print(f"    ✓ Keeping conversation ({conv_tokens} tokens, total: {current_conv_tokens})")
                    else:
                        print(f"    ✗ Dropping conversation ({conv_tokens} tokens, would exceed limit)")

            # Rebuild context with reduced conversations
            if conversations_to_keep:
                result = base_part + f"\\n\\nPrevious Conversations (Last {len(conversations_to_keep)}, reduced to fit tokens):\\n" + "\\n\\n---\\n\\n".join(conversations_to_keep)
            else:
                result = base_part  # No previous conversations fit, use only background
                print(f"    ⚠️ No previous conversations fit in token budget, using background only")
        else:
            result = base_part

        final_tokens = self.count_tokens(result)
        print(f"✂️ GPU {self.gpu_id}: Conversation reduction complete: {current_tokens} -> {final_tokens} tokens")
        return result

    def truncate_context_smartly(self, context: str, max_tokens: int) -> str:
        """Intelligently truncate context to fit within token limit - used as fallback when conversation reduction is not applicable"""
        current_tokens = self.count_tokens(context)

        if current_tokens <= max_tokens:
            return context

        print(f"⚠️ GPU {self.gpu_id}: Applying fallback smart truncation ({current_tokens} tokens) -> {max_tokens} tokens")

        # Split by sections and keep the most important parts
        sections = context.split('\n\n')

        # Always keep background if it exists
        background_parts = []
        cumulative_parts = []
        current_parts = []

        for section in sections:
            if 'Background Context:' in section or 'Case Background' in section:
                background_parts.append(section)
            elif 'Previous Conversations' in section or 'Previous Dialogue' in section:
                cumulative_parts.append(section)
            elif 'Current Conversation' in section:
                current_parts.append(section)
            else:
                background_parts.append(section)

        # Prioritize: Current > Background > Previous (most recent first)
        final_parts = []
        tokens_used = 0

        # Add current conversation context first (most important)
        for part in current_parts:
            part_tokens = self.count_tokens(part)
            if tokens_used + part_tokens <= max_tokens:
                final_parts.append(part)
                tokens_used += part_tokens

        # Add background context
        for part in background_parts:
            part_tokens = self.count_tokens(part)
            if tokens_used + part_tokens <= max_tokens:
                final_parts.append(part)
                tokens_used += part_tokens
            else:
                # Truncate this part
                remaining_tokens = max_tokens - tokens_used
                if remaining_tokens > 100:  # Only add if meaningful space left
                    truncated_part = part[:remaining_tokens * 4]  # Rough char to token ratio
                    actual_tokens = self.count_tokens(truncated_part)
                    if tokens_used + actual_tokens <= max_tokens:
                        final_parts.append(truncated_part + "...")
                        tokens_used += actual_tokens
                break

        # Add previous conversations (most recent first, but only if space)
        cumulative_parts.reverse()  # Most recent first
        for part in cumulative_parts:
            part_tokens = self.count_tokens(part)
            if tokens_used + part_tokens <= max_tokens:
                final_parts.append(part)
                tokens_used += part_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = max_tokens - tokens_used
                if remaining_tokens > 200:  # Only if significant space
                    truncated_part = part[:remaining_tokens * 4]
                    actual_tokens = self.count_tokens(truncated_part)
                    if tokens_used + actual_tokens <= max_tokens:
                        final_parts.append(truncated_part + "...")
                        tokens_used += actual_tokens
                break

        result = '\n\n'.join(final_parts)
        final_tokens = self.count_tokens(result)
        print(f"✂️ GPU {self.gpu_id}: Smart truncation complete: {current_tokens} -> {final_tokens} tokens")
        return result

    def generate_response(self, messages: List[Dict], max_tokens: int = 300, temperature: float = 0.1):
        """使用Qwen2.5-7B模型生成响应"""
        try:
            print(f"🔧 GPU {self.gpu_id} - 开始生成，max_tokens={max_tokens}, temperature={temperature}")

            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            print(f"📝 GPU {self.gpu_id} - 构建的prompt长度: {len(text)} 字符")

            # Use larger context for 72B model (more capable)
            model_max_length = 65536  # Qwen2.5-72B supports larger context
            reserved_tokens_for_generation = max_tokens + 200
            max_input_length = model_max_length - reserved_tokens_for_generation

            # Count actual tokens
            input_token_count = self.count_tokens(text)
            print(f"🔢 GPU {self.gpu_id} - 输入token数: {input_token_count} (max允许: {max_input_length})")

            # If input is too long, intelligently truncate
            if input_token_count > max_input_length:
                print(f"⚠️ GPU {self.gpu_id}: 输入过长，需要智能截断")
                # Extract user content from messages and truncate it
                for msg in messages:
                    if msg['role'] == 'user' and len(msg['content']) > 1000:
                        msg['content'] = self.truncate_context_smartly(msg['content'], max_input_length - 200)

                # Rebuild text after truncation
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                input_token_count = self.count_tokens(text)
                print(f"✂️ GPU {self.gpu_id}: 截断后token数: {input_token_count}")

            # Tokenize with final length check
            model_inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=max_input_length).to(self.device)

            # Ensure attention mask exists
            if model_inputs.attention_mask is None:
                model_inputs.attention_mask = torch.ones_like(model_inputs.input_ids)

            # Final safety check and hard truncation if needed
            actual_input_length = model_inputs.input_ids.shape[1]
            if actual_input_length > max_input_length:
                print(f"⚠️ GPU {self.gpu_id}: 最终硬截断 {actual_input_length} -> {max_input_length} tokens")
                model_inputs.input_ids = model_inputs.input_ids[:, -max_input_length:]
                model_inputs.attention_mask = model_inputs.attention_mask[:, -max_input_length:]
                actual_input_length = max_input_length

            # Calculate safe generation tokens
            max_generation_tokens = min(max_tokens, model_max_length - actual_input_length - 50)

            if max_generation_tokens < 20:
                print(f"❌ GPU {self.gpu_id}: 生成空间严重不足 ({max_generation_tokens} tokens)，强制设置为100")
                max_generation_tokens = 100  # Good balance for 7B model

            with torch.no_grad():
                # Clear GPU cache and check memory before inference
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    memory_before = torch.cuda.memory_allocated(self.gpu_id) / 1e9
                    print(f"⚡ GPU {self.gpu_id} - 开始模型推理... (内存使用: {memory_before:.2f}GB)")
                print(f"🎯 GPU {self.gpu_id} - 使用 {max_generation_tokens} tokens 用于生成 (输入: {actual_input_length}, 总预算: {model_max_length})")

                try:
                    generated_ids = self.model.generate(
                        model_inputs.input_ids,
                        attention_mask=model_inputs.attention_mask,
                        max_new_tokens=max_generation_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.8,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        repetition_penalty=1.1
                    )
                    print(f"⚡ GPU {self.gpu_id} - 模型推理完成")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"❌ GPU {self.gpu_id} - GPU OOM 错误，尝试更保守的设置")
                        torch.cuda.empty_cache()
                        # Try with smaller generation
                        max_generation_tokens = min(50, max_generation_tokens // 2)
                        generated_ids = self.model.generate(
                            model_inputs.input_ids,
                            attention_mask=model_inputs.attention_mask,
                            max_new_tokens=max_generation_tokens,
                            do_sample=False,  # Greedy decoding to save memory
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=False
                        )
                        print(f"⚡ GPU {self.gpu_id} - 使用保守设置完成推理 ({max_generation_tokens} tokens)")
                    else:
                        raise e

            print(f"🔢 GPU {self.gpu_id} - 生成的token数: {generated_ids.shape[1]}")
            print(f"🔢 GPU {self.gpu_id} - 新生成的token数: {generated_ids.shape[1] - model_inputs.input_ids.shape[1]}")

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"📄 GPU {self.gpu_id} - 完整response长度: {len(response)}")

            # 更好的提取方法：只解码新生成的token
            new_tokens = generated_ids[0][model_inputs.input_ids.shape[1]:]
            ai_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            print(f"📄 GPU {self.gpu_id} - 提取的ai_response长度: {len(ai_response)}")

            print("-" * 50)

            # 如果生成为空，额外调试信息
            if not ai_response:
                print(f"❌ GPU {self.gpu_id} - 生成为空！调试信息:")
                print(f"   原始prompt长度: {len(text)}")
                print(f"   完整response: {repr(response)}")
                print(f"   是否完整response等于prompt: {response == text}")
                print(f"   完整response后100字符: {repr(response[-100:])}")

            # 清理临时张量
            del model_inputs, generated_ids
            torch.cuda.empty_cache()

            return ai_response

        except Exception as e:
            print(f"❌ GPU {self.gpu_id} - 生成异常: {str(e)}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            return f"Error generating response on GPU {self.gpu_id}: {str(e)}"

    def get_qwen25_answer(self, messages: List[Dict], temperature: float = 0.1) -> str:
        """Get Qwen2.5's answer to the question"""
        try:
            response = self.generate_response(messages, max_tokens=2048, temperature=temperature)
            answer = response.strip().upper()
            import re
            print(f"🔍 Raw answer from model: {repr(response)}")

            # Extract just the letter if there's extra text
            if len(answer) > 1:
                """
                从 LLM 输出里提取 \boxed{...} 格式的答案
                """
                match = re.search(r"\\boxed\{(.*?)\}", response)
                if match:
                    return match.group(1).strip()

                # Look for "Answer: X" or "**Answer:** X" patterns first
                answer_pattern_match = re.search(r'[*\s]*answer[*\s]*:?\s*([ABCD])', response, re.IGNORECASE)
                if answer_pattern_match:
                    extracted_letter = answer_pattern_match.group(1).upper()
                    print(f"✅ Extracted from 'Answer:' pattern: {extracted_letter}")
                    return extracted_letter

                # If not in answer pattern, extract first standalone letter
                first_letter_match = re.search(r'\b([ABCD])\b', response)
                if first_letter_match:
                    extracted_letter = first_letter_match.group(1)
                    print(f"✅ Extracted standalone letter: {extracted_letter}")
                    return extracted_letter

            # If exact match
            if answer in ['A', 'B', 'C', 'D']:
                print(f"✅ Direct match answer: {answer}")
                return answer

            print(f"❌ No valid answer found in: {repr(answer)}")
            return 'INVALID'

        except Exception as e:
            print(f"Error getting Qwen2.5 response: {e}")
            return 'ERROR'

    def build_evaluation_prompt_no_leak(self, cumulative_context: str, question_data: Dict, target_justice: str = '') -> List[Dict]:
        """
        Build evaluation prompt for judicial intention analysis
        Uses cumulative context to understand judicial intentions
        """
        # Format the options nicely
        options_text = "\n".join([f"{key}: {value}" for key, value in question_data["options"].items()])

        messages = [
            {
                'role': 'system',
                'content': (
                    "You are an expert in legal reasoning and judicial intention analysis. "
                    "You will be given background context and a multiple-choice question "
                    "about understanding judicial intentions and reasoning patterns.\n\n"

                    "Your task is to:\n"
                    "1. Carefully analyze the judicial dialogue and reasoning\n"
                    "2. Consider the judge's legal philosophy and decision-making patterns\n"
                    "3. Evaluate the underlying intentions behind judicial questions and comments\n"
                    "4. Consider what legal principles or concerns are driving the judge's inquiries\n"
                    "5. Use your understanding of legal doctrine and judicial behavior\n"
                    "6. Use the background context to inform your analysis of judicial intentions\n"
                    "7. Select the most likely intention or reasoning from the given options\n\n"

                    "Focus on:\n"
                    "- Judicial reasoning patterns and legal philosophy\n"
                    "- Constitutional, statutory, or precedential concerns\n"
                    "- The judge's apparent legal priorities and interpretive approach\n"
                    "- Theory of mind understanding of judicial decision-making\n"
                    "- The underlying intentions driving judicial questions and comments\n\n"

                    "Respond with ONLY the letter of your chosen answer (A, B, C, or D). "
                    "Do not include any explanation or additional text."

                    "Provide brief one short sentence analysis and answer by selecting the options only"
                    "Please reiterate your answer, with your final answer a single answer of the form \\boxed{{answer}} at the end of your response."

                )
            },
            {
                'role': 'user',
                'content': (
                    f"Context (Case Background + Previous Conversations + Current Conversation Before Target Question):\n{cumulative_context}\n\n"
                    f"Target Justice: {target_justice}\n\n"
                    f"Question: {question_data['question']}\n\n"
                    f"Options:\n{options_text}\n\n"
                    f"What is your answer?"
                )
            }
        ]
        return messages

    def evaluate_dataset(self, input_file: str, output_file: str = None, delay: float = 1.0,
                        model: str = "qwen2.5-7b-instruct", use_cumulative_context: bool = True,
                        analyze_patterns: bool = True, include_current_conversation: bool = True,
                        use_summary: bool = False):
        """
        Evaluate the entire prediction dataset with cumulative context
        NO ANSWER LEAKAGE: Only shows context before target question
        """

        # Load the generated questions
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading input file: {e}")
            return

        base_background = data.get('background', '')
        conversations = data.get('conversations', [])

        print(f"开始使用{model}评估{len(conversations)}个对话...")
        print(f"基础背景长度: {len(base_background)} 字符")
        print(f"使用累积上下文: {use_cumulative_context}")
        print(f"包含当前对话: {include_current_conversation}")
        print(f"使用摘要压缩: {use_summary}")
        print(f"任务类型: 下一个问题预测 (无答案泄露)")
        print("-" * 60)

        # Sort conversations by ID to ensure proper order
        conversations.sort(key=lambda x: x.get('conversation_id', 0))

        # Track processed conversations for cumulative context
        processed_conversations = []

        # Process each conversation
        for conv in conversations:
            conv_id = conv.get('conversation_id', 0)
            dialogue = conv.get('dialogue', '')
            target_dialogue = conv.get('target_dialogue', dialogue)
            questions = conv.get('questions', [])

            print(f"\n处理对话 {conv_id}:")
            print(f"  问题数: {len(questions)}")

            # Process each question in the conversation
            for q_idx, question_data in enumerate(questions):
                if 'question' not in question_data or 'options' not in question_data:
                    print(f"    跳过格式错误的问题 {q_idx}")
                    continue

                # Build context with background and cumulative conversations
                context = ""
                if base_background:
                    if len(base_background) > 3000:
                        context = base_background[:3000] + "..."
                    else:
                        context = base_background

                # Add cumulative context with dynamic conversation reduction
                if use_cumulative_context:
                    current_conv_id = conv.get('conversation_id', 0)
                    previous_conversations_all = [c for c in conversations if c.get('conversation_id', 0) < current_conv_id]

                    # Sort by conversation_id to maintain order
                    previous_conversations_all.sort(key=lambda x: x.get('conversation_id', 0))

                    # Dynamically determine conversations to include based on token limits
                    max_context_tokens_before_current = 40000  # Large limit for 72B model
                    temp_context_with_prev = context
                    previous_dialogues_to_include = []

                    # Start with most recent conversations and add them one by one
                    for i in range(min(3, len(previous_conversations_all))):  # Up to 6 conversations for 7B model
                        prev_conv = previous_conversations_all[-(i+1)]
                        prev_full_dialogue = prev_conv.get('target_dialogue', prev_conv.get('dialogue', '')).strip()

                        if prev_full_dialogue:
                            # 使用摘要压缩对话 (如果启用)
                            if use_summary:
                                conv_id = prev_conv.get('conversation_id', 0)
                                # 检查是否已有缓存的摘要
                                if conv_id not in self.dialogue_summaries:
                                    print(f"    🔄 为对话 {conv_id} 生成摘要...")
                                    summary = self.summarize_dialogue(prev_full_dialogue, max_chars=3000)
                                    self.dialogue_summaries[conv_id] = summary
                                    print(f"    📝 摘要完成: {len(prev_full_dialogue)} -> {len(summary)} 字符")
                                else:
                                    print(f"    📋 使用缓存摘要 for 对话 {conv_id}")

                                dialogue_to_use = self.dialogue_summaries[conv_id]
                                context_type = "摘要"
                            else:
                                dialogue_to_use = prev_full_dialogue
                                context_type = "完整"

                            test_prev_dialogues = [dialogue_to_use] + previous_dialogues_to_include
                            test_context = temp_context_with_prev + f"\\n\\n以前的对话 (最后 {len(test_prev_dialogues)} 个):\\n" + "\\n\\n---\\n\\n".join(test_prev_dialogues)

                            test_tokens = self.count_tokens(test_context)
                            if test_tokens <= max_context_tokens_before_current:
                                previous_dialogues_to_include = test_prev_dialogues
                                print(f"    ✓ 添加了对话 {prev_conv.get('conversation_id', 0)} ({context_type}, 总token数: {test_tokens})")
                            else:
                                print(f"    ✗ 跳过对话 {prev_conv.get('conversation_id', 0)} (会超过token限制: {test_tokens})")
                                break

                    if previous_dialogues_to_include:
                        previous_dialogues_to_include.reverse()
                        summary_note = "摘要压缩" if use_summary else "完整"
                        context += f"\\n\\n以前的对话 (最后 {len(previous_dialogues_to_include)} 个, {summary_note}, 动态限制以适应token):\\n" + "\\n\\n---\\n\\n".join(previous_dialogues_to_include)
                        print(f"    添加了累积上下文 ({len(previous_dialogues_to_include)} 个以前的对话, {summary_note})")

                # Add current conversation context for intention analysis (NO ANSWER LEAKAGE)
                if include_current_conversation and dialogue:
                    # For motivation analysis, we need the dialogue before the current question
                    # to understand the context that led to the judge's intention
                    # We need to find where the current question appears and cut before it
                    current_question = question_data.get('question', '')

                    if current_question:
                        # Try to find the question in the dialogue to cut before it
                        question_snippet = current_question[:100]  # First 100 chars to find location
                        cutoff_position = dialogue.find(question_snippet)

                        if cutoff_position > 0:
                            safe_dialogue = dialogue[:cutoff_position].strip()
                            if safe_dialogue and current_question not in safe_dialogue:
                                context += f"\\n\\n当前对话 (问题前上下文):\\n{safe_dialogue}"
                                print(f"    安全: 添加了当前对话上下文 ({len(safe_dialogue)} 字符) - 在位置{cutoff_position}切断")
                            else:
                                print(f"    安全警告: 切断的对话可能包含答案信息，跳过当前对话")
                        else:
                            print(f"    警告: 无法在对话中定位当前问题，跳过当前对话上下文以避免答案泄露")
                    else:
                        print(f"    警告: 当前问题为空，跳过当前对话上下文")
                elif not include_current_conversation:
                    print(f"    跳过: include_current_conversation=False，不添加当前对话上下文")

                # Final context validation
                context_tokens = self.count_tokens(context)
                max_allowed_context_tokens = 50000  # Large limit for 72B model

                if context_tokens > max_allowed_context_tokens:
                    print(f"    ⚠️ 上下文过长 ({context_tokens} tokens)，应用对话减少")
                    context = self.reduce_conversations_to_fit_tokens(context, max_allowed_context_tokens, use_cumulative_context)
                    context_tokens = self.count_tokens(context)

                context_length = len(context)
                num_context_conversations = min(6, len(processed_conversations)) if use_cumulative_context else 0
                print(f"  问题 {q_idx + 1}/{len(questions)} (受限上下文: {num_context_conversations} 个以前的对话, 最多6个)")
                print(f"    上下文长度: {context_length} 字符, {context_tokens} tokens")

                # Build evaluation prompt for intention analysis
                messages = self.build_evaluation_prompt_no_leak(context, question_data)

                # Security validation: ensure no answer leakage
                prompt_text = str(messages).lower()
                correct_answer_text = question_data.get('debug_info', {}).get('correct_answer', '')

                # Check if any part of the correct answer appears in the prompt
                if correct_answer_text and correct_answer_text.lower() in prompt_text:
                    print(f"    ⚠️ 安全警告: 在对话{conv_id}的问题{q_idx}中检测到可能的答案泄露")

                # Additional security checks
                if 'debug_info' in prompt_text or 'correct_answer' in prompt_text:
                    raise ValueError(f"安全违规: 在对话{conv_id}的问题{q_idx}的评估提示中检测到答案泄露")

                qwen25_answer = self.get_qwen25_answer(messages)

                # Check if answer is correct
                correct_answer = question_data.get('answer', '').upper()
                is_correct = qwen25_answer == correct_answer

                # Extract target justice from the question text if available
                question_text = question_data.get('question', '')
                target_justice = ''
                if 'Justice' in question_text:
                    import re
                    justice_match = re.search(r'Justice (\w+)', question_text)
                    if justice_match:
                        target_justice = f"Justice {justice_match.group(1)}"

                result = {
                    "conversation_id": conv_id,
                    "question_index": q_idx,
                    "question": question_data['question'],
                    "target_justice": target_justice,
                    "correct_answer": correct_answer,
                    "qwen25_answer": qwen25_answer,
                    "is_correct": is_correct,
                    "options": question_data['options'],
                    "model_used": model,
                    "context_length": context_length,
                    "cumulative_conversations": min(3, len(processed_conversations)) if use_cumulative_context else 0,
                    "task_type": "intention_analysis",
                    "question_index_used": question_data.get('debug_info', {}).get('question_index_in_conversation', 0),
                    "security_check": "no_answer_leakage_verified"
                }

                self.results["detailed_results"].append(result)

                if result['is_correct']:
                    self.results["correct_answers"] += 1
                self.results["total_questions"] += 1

                print(f"    结果: {'✓' if is_correct else '✗'} (模型回答: {qwen25_answer}, 正确答案: {correct_answer})")

                # Add delay to avoid rate limits
                time.sleep(delay)

            # Add current conversation to processed list AFTER processing all its questions
            processed_conversations.append(conv)

        # Calculate final accuracy
        if self.results["total_questions"] > 0:
            self.results["accuracy"] = self.results["correct_answers"] / self.results["total_questions"]

        # Add configuration info to results
        self.results["model_used"] = model
        self.results["used_cumulative_context"] = use_cumulative_context
        self.results["include_current_conversation"] = include_current_conversation
        self.results["use_summary"] = use_summary
        self.results["analyzed_patterns"] = analyze_patterns
        self.results["dialogue_summaries"] = self.dialogue_summaries
        self.results["context_patterns"] = self.context_patterns

        # Print summary
        self.print_summary()

        # Save results
        if output_file:
            self.save_results(output_file)

    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("预测评估摘要 (Qwen2.5-7B模型)")
        print("="*60)
        print(f"任务类型: {self.results.get('task_type', '未知')}")
        print(f"使用的模型: {self.results.get('model_used', '未知')}")
        print(f"累积上下文: {self.results.get('used_cumulative_context', '未知')}")
        print(f"包含当前对话: {self.results.get('include_current_conversation', '未知')}")
        print(f"使用摘要压缩: {self.results.get('use_summary', '未知')}")
        print(f"模式分析: {self.results.get('analyzed_patterns', '未知')}")
        print(f"总问题数: {self.results['total_questions']}")
        print(f"正确预测数: {self.results['correct_answers']}")
        print(f"预测准确率: {self.results['accuracy']:.2%}")

        # Breakdown by conversation
        conv_stats = {}
        for result in self.results["detailed_results"]:
            conv_id = result["conversation_id"]
            if conv_id not in conv_stats:
                conv_stats[conv_id] = {"correct": 0, "total": 0}
            conv_stats[conv_id]["total"] += 1
            if result["is_correct"]:
                conv_stats[conv_id]["correct"] += 1

        print("\n按对话分解:")
        for conv_id in sorted(conv_stats.keys()):
            stats = conv_stats[conv_id]
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  对话 {conv_id}: {stats['correct']}/{stats['total']} ({accuracy:.1%})")

        # Context length analysis
        if self.results["detailed_results"]:
            context_lengths = [r.get("context_length", 0) for r in self.results["detailed_results"]]
            avg_context_length = sum(context_lengths) / len(context_lengths)
            print(f"\n平均上下文长度: {avg_context_length:.0f} 字符")

        # Answer distribution analysis
        answer_distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'ERROR': 0, 'INVALID': 0}
        for result in self.results["detailed_results"]:
            qwen25_answer = result["qwen25_answer"]
            if qwen25_answer in answer_distribution:
                answer_distribution[qwen25_answer] += 1

        print(f"\nQwen2.5-7B 答案分布:")
        for answer, count in answer_distribution.items():
            if count > 0:
                percentage = (count / self.results['total_questions']) * 100
                print(f"  {answer}: {count} ({percentage:.1f}%)")

    def save_results(self, output_file: str):
        """Save evaluation results to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存结果时出错: {e}")


def save_overall_accuracy_summary(output_dir: str, model_name: str, all_results: dict):
    """Save overall accuracy summary for all files and modes"""
    try:
        print(f"🔄 生成摘要: {model_name}")
        print(f"📊 all_results包含 {len(all_results)} 个条目")
        print(f"📝 all_results keys: {list(all_results.keys())}")

        summary_file = os.path.join(output_dir, f"{model_name}_overall_accuracy_summary.txt")

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"整体预测评估准确率摘要 ({model_name.upper()}模型)\n")
            f.write("=" * 80 + "\n")
            f.write(f"评估时间: {datetime.now().isoformat()}\n")
            f.write(f"评估文件数: {len(all_results)//2 if len(all_results) > 0 else 0}\n")  # 每个文件有2个版本
            f.write("\n")

            if not all_results:
                f.write("⚠️ 警告: 没有找到评估结果数据\n")
                f.write("可能的原因:\n")
                f.write("  - 评估过程中出现错误\n")
                f.write("  - 没有成功完成评估\n")
                f.write("  - 数据格式不匹配\n")
                print(f"⚠️ 警告: all_results为空，生成了基本摘要")
                return

            # Calculate overall statistics for each mode
            mode_stats = {
                "background_only": {"total": 0, "correct": 0, "files": []},
                "with_current": {"total": 0, "correct": 0, "files": []},
                "with_context": {"total": 0, "correct": 0, "files": []}
            }

            for result_key, result_data in all_results.items():
                print(f"📊 处理结果: {result_key}")
                if "background_only" in result_key:
                    mode_stats["background_only"]["total"] += result_data["total_questions"]
                    mode_stats["background_only"]["correct"] += result_data["correct_answers"]
                    mode_stats["background_only"]["files"].append({
                        "file": result_key,
                        "accuracy": result_data["accuracy"],
                        "total": result_data["total_questions"],
                        "correct": result_data["correct_answers"]
                    })
                elif "with_current" in result_key:
                    mode_stats["with_current"]["total"] += result_data["total_questions"]
                    mode_stats["with_current"]["correct"] += result_data["correct_answers"]
                    mode_stats["with_current"]["files"].append({
                        "file": result_key,
                        "accuracy": result_data["accuracy"],
                        "total": result_data["total_questions"],
                        "correct": result_data["correct_answers"]
                    })
                elif "with_context" in result_key:
                    mode_stats["with_context"]["total"] += result_data["total_questions"]
                    mode_stats["with_context"]["correct"] += result_data["correct_answers"]
                    mode_stats["with_context"]["files"].append({
                        "file": result_key,
                        "accuracy": result_data["accuracy"],
                        "total": result_data["total_questions"],
                        "correct": result_data["correct_answers"]
                    })

            # Write mode summaries
            f.write("模式对比:\n")
            f.write("-" * 60 + "\n")

            for mode_name, stats in mode_stats.items():
                if stats["total"] > 0:
                    overall_accuracy = stats["correct"] / stats["total"]
                    if mode_name == "background_only":
                        mode_display = "版本1: 仅背景"
                    elif mode_name == "with_current":
                        mode_display = "版本2: 背景+当前对话"
                    else:
                        mode_display = "版本3: 背景+前3个对话+当前对话"
                    f.write(f"{mode_display}:\n")
                    f.write(f"  总问题数: {stats['total']}\n")
                    f.write(f"  正确预测数: {stats['correct']}\n")
                    f.write(f"  整体准确率: {overall_accuracy:.4f} ({overall_accuracy:.2%})\n")
                    f.write("\n")

            # Detailed breakdown by file
            f.write("\n详细文件结果:\n")
            f.write("=" * 80 + "\n")

            for mode_name, stats in mode_stats.items():
                if stats["files"]:
                    if mode_name == "background_only":
                        mode_display = "版本1: 仅背景"
                    elif mode_name == "with_current":
                        mode_display = "版本2: 背景+当前对话"
                    else:
                        mode_display = "版本3: 背景+前3个对话+当前对话"
                    f.write(f"\n{mode_display}:\n")
                    f.write("-" * 60 + "\n")

                    for file_result in stats["files"]:
                        file_name = file_result["file"].replace("eval_", "").replace(f"_{model_name.replace('-', '_')}.json", "")
                        f.write(f"  {file_name}:\n")
                        f.write(f"    准确率: {file_result['accuracy']:.4f} ({file_result['accuracy']:.2%})\n")
                        f.write(f"    结果: {file_result['correct']}/{file_result['total']}\n")
                        f.write("\n")

        print(f"✅ 整体准确率摘要已保存到: {summary_file}")
    except Exception as e:
        print(f"❌ 保存整体准确率摘要时出错: {e}")
        import traceback
        traceback.print_exc()

        # 创建一个基本的错误摘要文件
        try:
            error_summary_file = os.path.join(output_dir, f"{model_name}_overall_accuracy_summary.txt")
            with open(error_summary_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"整体预测评估准确率摘要 ({model_name.upper()}模型)\n")
                f.write("=" * 80 + "\n")
                f.write(f"评估时间: {datetime.now().isoformat()}\n")
                f.write(f"状态: 错误 - {str(e)}\n")
                f.write("\n")
                f.write("❌ 摘要生成过程中发生错误\n")
                f.write(f"错误详情: {str(e)}\n")
                f.write("请检查评估脚本和数据格式\n")
        except:
            pass
def main():
    """Main evaluation function for intention task using Qwen2.5-72B model"""

    # Configuration
    input_dir = os.path.join(project_root, "data", "output_motivation")
    output_dir = os.path.join(project_root, "result_mode", "intention_result", "qwen2.5_72b_instruct")
    os.makedirs(output_dir, exist_ok=True)

    delay_between_requests = 1.0  # seconds

    # Using Qwen2.5-72B-Instruct model
    evaluation_model = "qwen2.5-72b-instruct"

    # Dictionary to store all results for overall summary
    all_results = {}

    # Get all JSON files from input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    if not json_files:
        print(f"在{input_dir}中没有找到JSON文件")
        return

    print(f"找到{len(json_files)}个要评估的JSON文件:")
    for f in json_files:
        print(f"  - {f}")

    print(f"\\n开始使用{evaluation_model}进行评估...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"为每个文件生成2个版本: 仅背景 & 带上下文 (限制为最后6个对话)")
    print("-" * 60)

    # Create single evaluator instance to reuse the loaded model
    print("🚀 一次性加载Qwen2.5-72B模型用于所有评估...")
    evaluator = IntentionEvaluator()
    print("✅ 72B模型加载成功，继续进行评估...")

    # Process each file with TWO different context settings for comparison
    for json_file in json_files:
        print(f"\\n{'='*60}")
        print(f"评估: {json_file}")
        print(f"{'='*60}")

        input_file = os.path.join(input_dir, json_file)
        base_name = os.path.splitext(json_file)[0]

        # VERSION 1: Background + Question (no cumulative context, no current conversation)
        print(f"\\n--- 版本 1: 背景 + 问题 (无累积上下文, 无当前对话) ---")
        output_file_bg = os.path.join(output_dir, f"eval_{base_name}_background_only_{evaluation_model.replace('-', '_')}.json")

        # Check if file already exists
        if os.path.exists(output_file_bg):
            print(f"⏭️  版本1文件已存在，跳过评估: {os.path.basename(output_file_bg)}")
            try:
                with open(output_file_bg, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    bg_result_key = f"eval_{base_name}_background_only_{evaluation_model.replace('-', '_')}.json"
                    all_results[bg_result_key] = {
                        "total_questions": existing_data.get("total_questions", 0),
                        "correct_answers": existing_data.get("correct_answers", 0),
                        "accuracy": existing_data.get("accuracy", 0.0)
                    }
                    print(f"✅ 已加载现有结果: {existing_data.get('total_questions', 0)}题, 准确率{existing_data.get('accuracy', 0.0):.2%}")
            except Exception as e:
                print(f"⚠️  加载现有文件失败: {e}，将重新评估")
        else:
            # Reset results for new evaluation but keep the loaded model
            evaluator.results = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "total_questions": 0,
                "correct_answers": 0,
                "accuracy": 0.0,
                "detailed_results": [],
                "task_type": "intention_analysis"
            }
            evaluator.dialogue_summaries = {}
            evaluator.context_patterns = {}
    
            evaluator.evaluate_dataset(
                input_file=input_file,
                output_file=output_file_bg,
                delay=delay_between_requests,
                model=evaluation_model,
                use_cumulative_context=False,
                analyze_patterns=False,
                include_current_conversation=False
            )
    
            # Store results for overall summary
            bg_result_key = f"eval_{base_name}_background_only_{evaluation_model.replace('-', '_')}.json"
            all_results[bg_result_key] = {
                "total_questions": evaluator.results["total_questions"],
                "correct_answers": evaluator.results["correct_answers"],
                "accuracy": evaluator.results["accuracy"]
            }

        # VERSION 2: Background + current conversation (no cumulative context)
        print(f"\\n--- 版本 2: 背景 + 当前对话 (无累积上下文) ---")
        output_file_current = os.path.join(output_dir, f"eval_{base_name}_with_current_{evaluation_model.replace('-', '_')}.json")

        # Check if file already exists
        if os.path.exists(output_file_current):
            print(f"⏭️  版本2文件已存在，跳过评估: {os.path.basename(output_file_current)}")
            try:
                with open(output_file_current, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    current_result_key = f"eval_{base_name}_with_current_{evaluation_model.replace('-', '_')}.json"
                    all_results[current_result_key] = {
                        "total_questions": existing_data.get("total_questions", 0),
                        "correct_answers": existing_data.get("correct_answers", 0),
                        "accuracy": existing_data.get("accuracy", 0.0)
                    }
                    print(f"✅ 已加载现有结果: {existing_data.get('total_questions', 0)}题, 准确率{existing_data.get('accuracy', 0.0):.2%}")
            except Exception as e:
                print(f"⚠️  加载现有文件失败: {e}，将重新评估")
        else:
            # Reset results for new evaluation but keep the loaded model
            evaluator.results = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "total_questions": 0,
                "correct_answers": 0,
                "accuracy": 0.0,
                "detailed_results": [],
                "task_type": "intention_analysis"
            }
            evaluator.dialogue_summaries = {}
            evaluator.context_patterns = {}
    
            evaluator.evaluate_dataset(
                input_file=input_file,
                output_file=output_file_current,
                delay=delay_between_requests,
                model=evaluation_model,
                use_cumulative_context=False,
                analyze_patterns=False,
                include_current_conversation=True,
                use_summary=False
            )
    
            # Store results for overall summary
            current_result_key = f"eval_{base_name}_with_current_{evaluation_model.replace('-', '_')}.json"
            all_results[current_result_key] = {
                "total_questions": evaluator.results["total_questions"],
                "correct_answers": evaluator.results["correct_answers"],
                "accuracy": evaluator.results["accuracy"]
            }

        # VERSION 3: Background + last 3 conversations + current conversation + question
        print(f"\\n--- 版本 3: 背景 + 前面3个对话 + 当前对话 + 问题 ---")
        output_file_full = os.path.join(output_dir, f"eval_{base_name}_with_context_{evaluation_model.replace('-', '_')}.json")

        # Check if file already exists
        if os.path.exists(output_file_full):
            print(f"⏭️  版本3文件已存在，跳过评估: {os.path.basename(output_file_full)}")
            try:
                with open(output_file_full, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    full_result_key = f"eval_{base_name}_with_context_{evaluation_model.replace('-', '_')}.json"
                    all_results[full_result_key] = {
                        "total_questions": existing_data.get("total_questions", 0),
                        "correct_answers": existing_data.get("correct_answers", 0),
                        "accuracy": existing_data.get("accuracy", 0.0)
                    }
                    print(f"✅ 已加载现有结果: {existing_data.get('total_questions', 0)}题, 准确率{existing_data.get('accuracy', 0.0):.2%}")
            except Exception as e:
                print(f"⚠️  加载现有文件失败: {e}，将重新评估")
        else:
            # Reset results for new evaluation but keep the loaded model
            evaluator.results = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "total_questions": 0,
                "correct_answers": 0,
                "accuracy": 0.0,
                "detailed_results": [],
                "task_type": "intention_analysis"
            }
            evaluator.dialogue_summaries = {}
            evaluator.context_patterns = {}
    
            evaluator.evaluate_dataset(
                input_file=input_file,
                output_file=output_file_full,
                delay=delay_between_requests,
                model=evaluation_model,
                use_cumulative_context=True,
                analyze_patterns=False,
                include_current_conversation=True,
                use_summary=False
            )
    
            # Store results for overall summary
            full_result_key = f"eval_{base_name}_with_context_{evaluation_model.replace('-', '_')}.json"
            all_results[full_result_key] = {
                "total_questions": evaluator.results["total_questions"],
                "correct_answers": evaluator.results["correct_answers"],
                "accuracy": evaluator.results["accuracy"]
            }

        print(f"{json_file}的评估完成 (生成了3个版本)")

    print(f"\\n{'='*60}")
    print("所有评估完成")
    print(f"{'='*60}")
    print(f"结果保存在: {output_dir}")
    print(f"评估的文件: {len(json_files)} 个文件 × 每个3个版本")

    # Generate overall accuracy summary
    save_overall_accuracy_summary(output_dir, evaluation_model, all_results)

if __name__ == "__main__":
    main()