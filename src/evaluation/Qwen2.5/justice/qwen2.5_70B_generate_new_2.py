# Get project root directory
import os
# This file is at: src/evaluators/.../generate/*.py
# Need to go up 5 levels to reach project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Fix Windows encoding issue
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

class JusticeKeywordEvaluator:
    def __init__(self):
        # Qwen2.5-72B-Instruct local model configuration
        model_path = os.path.join(project_root, "qwen_models", "Qwen_Qwen2.5-72B-Instruct")
        self.model_name = model_path

        print("🔄 Loading Qwen2.5-72B-Instruct model...")
        print(f"   Model path: {model_path}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {self.device}")

        # Load model with appropriate settings
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use half precision for GPU
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use full precision for CPU
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)

        self.model.eval()  # Set to evaluation mode
        print("✅ Model loaded successfully!")

        self.results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "detailed_results": [],
            "task_type": "justice_keyword_identification"
        }
        self.dialogue_summaries = {}
        self.context_patterns = {}


    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the actual tokenizer"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def reduce_conversations_to_fit_tokens(self, context: str, max_tokens: int, use_cumulative_context: bool = True) -> str:
        """Reduce number of conversations instead of hard truncation to fit token limit"""
        current_tokens = self.count_tokens(context)

        if current_tokens <= max_tokens:
            return context

        print(f"⚠️ Context too long ({current_tokens} tokens), reducing conversations to fit {max_tokens} tokens")

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
        print(f"✂️ Conversation reduction complete: {current_tokens} -> {final_tokens} tokens")
        return result

    def truncate_context_smartly(self, context: str, max_tokens: int) -> str:
        """Intelligently truncate context to fit within token limit - used as fallback when conversation reduction is not applicable"""
        current_tokens = self.count_tokens(context)

        if current_tokens <= max_tokens:
            return context

        print(f"⚠️ Applying fallback smart truncation ({current_tokens} tokens) -> {max_tokens} tokens")

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
        print(f"✂️ Smart truncation complete: {current_tokens} -> {final_tokens} tokens")
        return result

    def generate_response(self, messages: List[Dict], max_tokens: int = 300):
        """Generate response using local Qwen2.5-72B model"""
        try:
            print(f"🔧 Starting generation with Qwen2.5-72B, max_tokens={max_tokens}")

            # Format messages using Qwen's chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize and generate
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Extract only the generated part (remove input prompt)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            ai_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"📄 Raw response: {repr(ai_response)}")

            if ai_response is None:
                ai_response = ""
            else:
                ai_response = ai_response.strip()

            print(f"📄 Generated response length: {len(ai_response)} characters")
            print("-" * 50)

            return ai_response

        except Exception as e:
            print(f"❌ Generation error: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error generating response: {str(e)}"


    def summarize_dialogue(self, dialogue: str, max_chars: int = 4000) -> str:
        """
        Generate concise summary of dialogue using Qwen2.5
        """
        try:
            if len(dialogue) <= max_chars:
                return dialogue

            # If dialogue is short, return directly
            if len(dialogue) <= 5000:
                return dialogue[:max_chars] + "..."

            messages = [
                {
                    'role': 'system',
                    'content': (
                        "You are a legal dialogue summary expert. Please summarize the given court dialogue into a concise summary, "
                        f"limited to {max_chars} characters. Preserve key legal issues, main concerns of judges, "
                        "and core arguments of lawyers. Use objective, accurate language."
                    )
                },
                {
                    'role': 'user',
                    'content': f"Please summarize the following court dialogue (limit {max_chars} characters):\n\n{dialogue}"
                }
            ]

            summary = self.generate_response(messages, max_tokens=2000)

            # Ensure summary doesn't exceed character limit
            if len(summary) > max_chars:
                summary = summary[:max_chars] + "..."

            return summary.strip()

        except Exception as e:
            print(f"Error generating summary: {e}")
            # If summary fails, use simple truncation
            return dialogue[:max_chars] + "..." if len(dialogue) > max_chars else dialogue

    def get_qwen_answer(self, messages: List[Dict]) -> str:
        """Get Qwen2.5's answer to the question"""
        try:
            response = self.generate_response(messages, max_tokens=3072)
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

                # If not in box, extract first uppercase letter
                first_letter_match = re.search(r'[ABCD]', response)
                if first_letter_match:
                    extracted_letter = first_letter_match.group(0)
                    print(f"✅ Extracted first letter: {extracted_letter}")
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

    def build_evaluation_prompt_no_leak(self, cumulative_context: str, question_data: Dict) -> List[Dict]:
        """
        Build evaluation prompt for advocate keyword identification
        Uses cumulative context for better understanding
        SECURITY: Does not include original statement to prevent answer leakage
        """
        # Format the options nicely
        options_text = "\n".join([f"{key}: {value}" for key, value in question_data["options"].items()])

        # Get speaker info but NOT the original statement (contains answer)
        context_info = question_data.get('metadata', {}).get('context_info', {})
        speaker = context_info.get('speaker', '')

        # SECURITY: Do not include original_statement as it contains the correct answer

        messages = [
            {
                'role': 'system',
                'content': (
                    "You are an expert in legal language analysis and judicial questioning comprehension. "
                    "You will identify key legal terms, phrases, or concepts from judicial questions.\n\n"

                    "Your task is to:\n"
                    "1. Analyze the case context and dialogue background\n"
                    "2. Focus on the specific judicial question provided\n"
                    "3. Identify which legal concept, term, or phrase is most relevant to the justice's inquiry\n"
                    "4. Consider the legal significance and context of each option\n"
                    "5. Select the option that best captures the key legal point being made\n\n"

                    "Focus on:\n"
                    "- Core legal concepts and principles\n"
                    "- Important statutory references and legal authorities\n"
                    "- Procedural requirements and legal standards\n"
                    "- Case law principles and precedents\n"
                    "- Constitutional provisions and interpretations\n"
                    "- Technical legal terminology essential to the argument\n\n"

                    "Provide brief one short sentence analysis and answer by selecting the options only"
                    "Please reiterate your answer, with your final answer a single answer of the form \\boxed{{answer}} at the end of your response."
                )
            },
            {
                'role': 'user',
                'content': (
                    f"Case Context:\n{cumulative_context}\n\n"
                    f"Question: {question_data['question']}\n\n"
                    f"Options:\n{options_text}\n\n"
                    f"Based on the context provided, what is your answer?"
                )
            }
        ]
        return messages

    def evaluate_dataset(self, input_file: str, output_file: str = None, delay: float = 1.0,
                        model: str = "qwen2.5-72b-instruct", use_cumulative_context: bool = True,
                        analyze_patterns: bool = True, include_current_conversation: bool = True,
                        use_summary: bool = False):
        """
        Evaluate the advocate keyword identification dataset
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
        print(f"任务类型: 法官关键词识别")
        print("-" * 60)

        # Sort conversations by ID to ensure proper order
        conversations.sort(key=lambda x: x.get('conversation_id', 0))

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
                    if len(base_background) > 4000:
                        context = base_background[:4000] + "..."
                    else:
                        context = base_background

                # Add cumulative context (previous conversations) if enabled
                # SECURITY: These are complete previous conversations, should not contain current question's answer
                if use_cumulative_context:
                    current_conv_id = conv.get('conversation_id', 0)
                    previous_conversations_all = [c for c in conversations if c.get('conversation_id', 0) < current_conv_id]

                    # Sort by conversation_id to maintain order and take last 3
                    previous_conversations_all.sort(key=lambda x: x.get('conversation_id', 0))
                    previous_conversations_to_include = previous_conversations_all[-3:] if len(previous_conversations_all) > 3 else previous_conversations_all

                    if previous_conversations_to_include:
                        previous_dialogues = []
                        correct_phrase = question_data.get('metadata', {}).get('correct_phrase', '')

                        for prev_conv in previous_conversations_to_include:
                            prev_dialogue = prev_conv.get('dialogue', '').strip()
                            if prev_dialogue:
                                # Security check: ensure the current question's answer is not in previous conversations
                                # Use more sophisticated matching - only flag if phrase appears as complete words
                                if correct_phrase and len(correct_phrase) > 3:  # Only check phrases longer than 3 chars
                                    # Create word boundary pattern to avoid partial matches
                                    pattern = r'\b' + re.escape(correct_phrase.lower()) + r'\b'
                                    if re.search(pattern, prev_dialogue.lower()):
                                        print(f"    ⚠️ 安全警告: 前面对话 {prev_conv.get('conversation_id', 0)} 包含当前问题的答案，跳过")
                                        continue
                                previous_dialogues.append(prev_dialogue)

                        if previous_dialogues:
                            context += f"\n\n前面的对话 (最后 {len(previous_dialogues)} 个, 已验证无答案泄露):\n" + "\n\n---\n\n".join(previous_dialogues)
                            print(f"    ✅ 安全添加了累积上下文 ({len(previous_dialogues)} 个前面的对话, 已过滤答案泄露)")

                # Add current conversation context BEFORE the target statement (NO ANSWER LEAKAGE)
                if include_current_conversation:
                    dialogue_content = conv.get('dialogue', '')
                    if dialogue_content:
                        # Get the original statement that contains the correct answer
                        context_info = question_data.get('metadata', {}).get('context_info', {})
                        original_statement = context_info.get('original_statement', '')

                        # 🔍 DEBUG: Check data structure
                        print(f"    🔍 Debug - metadata keys: {list(question_data.get('metadata', {}).keys())}")
                        print(f"    🔍 Debug - context_info keys: {list(context_info.keys())}")
                        print(f"    🔍 Debug - original_statement exists: {bool(original_statement)}")
                        if original_statement:
                            print(f"    🔍 Debug - original_statement length: {len(original_statement)} chars")
                            print(f"    🔍 Debug - dialogue_content length: {len(dialogue_content)} chars")

                        if original_statement:
                            # 🔪 PRECISE CUTOFF: Use complete original_statement for exact matching
                            print(f"    🔍 尝试精确匹配完整原始陈述...")
                            cutoff_position = dialogue_content.find(original_statement)

                            if cutoff_position != -1:
                                print(f"    🎯 ✅ 找到完整原始陈述在位置 {cutoff_position}")
                                print(f"    📝 原始陈述开头: '{original_statement[:50]}...'")
                                print(f"    📝 匹配处的对话: '{dialogue_content[cutoff_position:cutoff_position+50]}...'")
                            else:
                                print(f"    ❌ 完整原始陈述未找到，尝试部分匹配...")
                                # Fallback: try to find substantial part of the statement
                                original_words = original_statement.split()
                                if len(original_words) >= 10:
                                    # Try first 10 words as a fallback
                                    partial_statement = ' '.join(original_words[:10])
                                    print(f"    🔍 尝试匹配前10个词: '{partial_statement}'")
                                    cutoff_position = dialogue_content.find(partial_statement)
                                    if cutoff_position != -1:
                                        print(f"    🎯 ✅ 找到部分原始陈述在位置 {cutoff_position} (前10个词)")
                                        print(f"    📝 匹配处的对话: '{dialogue_content[cutoff_position:cutoff_position+50]}...'")
                                    else:
                                        print(f"    ❌ 无法在对话中找到原始陈述的任何部分")
                                        print(f"    📝 寻找的陈述: '{original_statement[:100]}...'")
                                        print(f"    📝 对话开头: '{dialogue_content[:200]}...'")
                                        print(f"    📝 对话结尾: '...{dialogue_content[-200:]}'" if len(dialogue_content) > 200 else "")
                                elif len(original_words) >= 5:
                                    # Try first 5 words for very short statements
                                    partial_statement = ' '.join(original_words[:5])
                                    print(f"    🔍 尝试匹配前5个词: '{partial_statement}'")
                                    cutoff_position = dialogue_content.find(partial_statement)
                                    if cutoff_position != -1:
                                        print(f"    🎯 ✅ 找到部分原始陈述在位置 {cutoff_position} (前5个词)")
                                    else:
                                        print(f"    ❌ 前5个词也未找到")
                                else:
                                    print(f"    ❌ 原始陈述太短({len(original_words)} 词)，无法可靠匹配")

                            if cutoff_position > 0:  # Ensure there's some content before the cutoff
                                # 🔪 HARD CUT: Include only dialogue BEFORE original_statement starts
                                safe_dialogue = dialogue_content[:cutoff_position].strip()

                                print(f"    🔪 截断结果验证:")
                                print(f"    📏 原始对话长度: {len(dialogue_content)} 字符")
                                print(f"    📏 截断位置: {cutoff_position}")
                                print(f"    📏 截断后长度: {len(safe_dialogue)} 字符")
                                print(f"    📝 截断后对话结尾: '...{safe_dialogue[-100:]}'" if len(safe_dialogue) > 100 else f"    📝 截断后完整对话: '{safe_dialogue}'")

                                # Verify that original_statement is NOT in the safe_dialogue
                                if original_statement in safe_dialogue:
                                    print(f"    🚨 警告: original_statement仍然存在于截断后的对话中！")
                                else:
                                    print(f"    ✅ 验证通过: original_statement不在截断后的对话中")

                                if safe_dialogue and len(safe_dialogue) > 50:  # Ensure meaningful content
                                    # Since we cut at the exact start of original_statement, no additional checks needed
                                    context += f"\n\n当前对话 (在original_statement之前截断):\n{safe_dialogue}"
                                    print(f"    ✅ 硬截断成功: 在original_statement开始位置 {cutoff_position} 处截断")
                                    print(f"    📏 最终安全对话长度: {len(safe_dialogue)} 字符")
                                else:
                                    print(f"    ⚠️ 截断后的上文太短 ({len(safe_dialogue) if safe_dialogue else 0} 字符), 跳过")
                            elif cutoff_position == 0:
                                print(f"    ⚠️ original_statement在对话开头 (位置0)，没有可用的上文")
                            else:
                                print(f"    ❌ 未找到original_statement，无法截断")
                        else:
                            print(f"    ⚠️ 缺少original_statement，跳过当前对话上下文以确保安全")
                    else:
                        print(f"    警告: 当前对话内容为空")

                # Final context validation
                context_tokens = self.count_tokens(context)
                max_allowed_context_tokens = 100000  # Very large limit for 72B model

                if context_tokens > max_allowed_context_tokens:
                    print(f"    ⚠️ 上下文过长 ({context_tokens} tokens)，应用对话减少")
                    context = self.reduce_conversations_to_fit_tokens(context, max_allowed_context_tokens, use_cumulative_context)
                    context_tokens = self.count_tokens(context)

                context_length = len(context)
                print(f"  问题 {q_idx + 1}/{len(questions)}")
                print(f"    上下文长度: {context_length} 字符, {context_tokens} tokens")

                # Build evaluation prompt
                messages = self.build_evaluation_prompt_no_leak(context, question_data)

                # Final security check: ensure no answer leakage in CONTEXT (not in question itself)
                # For fill-in-blank questions, the answer phrase may legitimately appear in the question text
                correct_phrase = question_data.get('metadata', {}).get('correct_phrase', '')
                if correct_phrase and len(correct_phrase) > 3:  # Only check phrases longer than 3 chars
                    # Only check the CONTEXT part, not the question itself
                    context_only = context.lower()  # Only check the context we built, not the question
                    pattern = r'\b' + re.escape(correct_phrase.lower()) + r'\b'
                    if re.search(pattern, context_only):
                        print(f"    🚨 严重安全警告: 检测到答案泄露在上下文中! 答案: '{correct_phrase}'")
                        print(f"    跳过此问题以防止答案泄露")
                        continue
                    else:
                        print(f"    ✅ 安全检查通过: 答案 '{correct_phrase}' 不在上下文中 (可能在问题文本中，这是正常的)")
                else:
                    print(f"    ✅ 跳过安全检查: 短语过短或不存在")

                qwen_answer = self.get_qwen_answer(messages)

                # Check if answer is correct
                correct_answer = question_data.get('answer', '').upper()
                is_correct = qwen_answer == correct_answer

                # Get metadata for this question
                metadata = question_data.get('metadata', {})
                context_info = metadata.get('context_info', {})

                result = {
                    "conversation_id": conv_id,
                    "question_index": q_idx,
                    "question": question_data['question'],
                    "correct_answer": correct_answer,
                    "qwen_answer": qwen_answer,
                    "is_correct": is_correct,
                    "options": question_data['options'],
                    "model_used": model,
                    "context_length": context_length,
                    "task_type": "justice_keyword_identification",
                    "metadata": {
                        "correct_phrase": metadata.get('correct_phrase', ''),
                        "category": metadata.get('category', ''),
                        "difficulty": metadata.get('difficulty', ''),
                        "word_count": metadata.get('word_count', 0),
                        "speaker": context_info.get('speaker', ''),
                        "explanation": metadata.get('explanation', '')
                    }
                }

                self.results["detailed_results"].append(result)

                if result['is_correct']:
                    self.results["correct_answers"] += 1
                self.results["total_questions"] += 1

                correct_phrase = metadata.get('correct_phrase', '')
                print(f"    结果: {'✓' if is_correct else '✗'} (预测: {qwen_answer}, 正确: {correct_answer}) - 关键词: {correct_phrase}")

                # Add delay to avoid rate limits
                time.sleep(delay)

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
        print("法官关键词识别评估摘要 (Qwen2.5模型)")
        print("="*60)
        print(f"任务类型: {self.results.get('task_type', '未知')}")
        print(f"使用的模型: {self.results.get('model_used', '未知')}")
        print(f"累积上下文: {self.results.get('used_cumulative_context', '未知')}")
        print(f"包含当前对话: {self.results.get('include_current_conversation', '未知')}")
        print(f"使用摘要压缩: {self.results.get('use_summary', '未知')}")
        print(f"模式分析: {self.results.get('analyzed_patterns', '未知')}")
        print(f"总问题数: {self.results['total_questions']}")
        print(f"正确识别数: {self.results['correct_answers']}")
        print(f"识别准确率: {self.results['accuracy']:.2%}")

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
            qwen_answer = result["qwen_answer"]
            if qwen_answer in answer_distribution:
                answer_distribution[qwen_answer] += 1

        print(f"\nQwen2.5 答案分布:")
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
        summary_file = os.path.join(output_dir, f"{model_name}_overall_accuracy_summary.txt")

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"整体法官关键词识别准确率摘要 ({model_name.upper()}模型)\n")
            f.write("=" * 80 + "\n")
            f.write(f"评估时间: {datetime.now().isoformat()}\n")
            f.write(f"评估文件数: {len(all_results)//3}\n")  # 每个文件有3个模式
            f.write("\n")

            # Calculate overall statistics for each mode
            mode_stats = {
                "mode1": {"total": 0, "correct": 0, "files": []},
                "mode2": {"total": 0, "correct": 0, "files": []},
                "mode3": {"total": 0, "correct": 0, "files": []}
            }

            for result_key, result_data in all_results.items():
                if "mode1" in result_key:
                    mode_stats["mode1"]["total"] += result_data["total_questions"]
                    mode_stats["mode1"]["correct"] += result_data["correct_answers"]
                    mode_stats["mode1"]["files"].append({
                        "file": result_key,
                        "accuracy": result_data["accuracy"],
                        "total": result_data["total_questions"],
                        "correct": result_data["correct_answers"]
                    })
                elif "mode2" in result_key:
                    mode_stats["mode2"]["total"] += result_data["total_questions"]
                    mode_stats["mode2"]["correct"] += result_data["correct_answers"]
                    mode_stats["mode2"]["files"].append({
                        "file": result_key,
                        "accuracy": result_data["accuracy"],
                        "total": result_data["total_questions"],
                        "correct": result_data["correct_answers"]
                    })
                elif "mode3" in result_key:
                    mode_stats["mode3"]["total"] += result_data["total_questions"]
                    mode_stats["mode3"]["correct"] += result_data["correct_answers"]
                    mode_stats["mode3"]["files"].append({
                        "file": result_key,
                        "accuracy": result_data["accuracy"],
                        "total": result_data["total_questions"],
                        "correct": result_data["correct_answers"]
                    })

            # Write mode summaries
            f.write("模式对比:\n")
            f.write("-" * 60 + "\n")

            mode_descriptions = {
                "mode1": "Mode 1: 仅背景",
                "mode2": "Mode 2: 背景+当前对话(截断)",
                "mode3": "Mode 3: 背景+当前对话(截断)+前3个对话"
            }

            for mode_name, stats in mode_stats.items():
                if stats["total"] > 0:
                    overall_accuracy = stats["correct"] / stats["total"]
                    mode_display = mode_descriptions.get(mode_name, mode_name)
                    f.write(f"{mode_display}:\n")
                    f.write(f"  总问题数: {stats['total']}\n")
                    f.write(f"  正确识别数: {stats['correct']}\n")
                    f.write(f"  整体准确率: {overall_accuracy:.4f} ({overall_accuracy:.2%})\n")
                    f.write("\n")

            # Detailed breakdown by file
            f.write("\n详细文件结果:\n")
            f.write("=" * 80 + "\n")

            for mode_name, stats in mode_stats.items():
                if stats["files"]:
                    mode_display = mode_descriptions.get(mode_name, mode_name)
                    f.write(f"\n{mode_display}:\n")
                    f.write("-" * 60 + "\n")

                    for file_result in stats["files"]:
                        file_name = file_result["file"].replace("eval_", "").replace(f"_{model_name.replace('-', '_')}.json", "")
                        f.write(f"  {file_name}:\n")
                        f.write(f"    准确率: {file_result['accuracy']:.4f} ({file_result['accuracy']:.2%})\n")
                        f.write(f"    结果: {file_result['correct']}/{file_result['total']}\n")
                        f.write("\n")

        print(f"\n整体准确率摘要已保存到: {summary_file}")
    except Exception as e:
        print(f"保存整体准确率摘要时出错: {e}")

def main():
    """Main evaluation function for justice keyword identification using Qwen2.5-70B"""

    # Configuration
    input_dir = os.path.join(project_root, "data", "generate_justice_questions")
    output_dir = os.path.join(project_root, "result_mode", "generate_result_justice", "qwen2.5_72b_instruct")
    os.makedirs(output_dir, exist_ok=True)

    delay_between_requests = 0.0  # No delay needed for local model

    # Using Qwen2.5-72B local model
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
    print(f"任务类型: 法官关键词识别")
    print(f"为每个文件生成3个模式: 背景 & 背景+当前对话 & 背景+前3个对话+当前对话")
    print("-" * 60)

    # Create single evaluator instance
    print("🚀 Initializing Qwen2.5 evaluator...")
    evaluator = JusticeKeywordEvaluator()
    print("✅ Qwen2.5 evaluator initialized successfully...")

    # Process each file with TWO different context settings for comparison
    for json_file in json_files:
        print(f"\\n{'='*60}")
        print(f"评估: {json_file}")
        print(f"{'='*60}")

        input_file = os.path.join(input_dir, json_file)
        base_name = os.path.splitext(json_file)[0]

        # Clean the base name to remove "justice_keywords_" prefix if present
        if base_name.startswith("justice_keywords_"):
            base_name = base_name[18:]  # Remove "justice_keywords_" prefix

        # MODE 1: Background only (no conversations)
        print(f"\\n--- Mode 1: 仅背景 (无对话) ---")
        output_file_mode1 = os.path.join(output_dir, f"eval_{base_name}_mode1_background_only_{evaluation_model.replace('-', '_')}.json")

        # Reset results for new evaluation
        evaluator.results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "detailed_results": [],
            "task_type": "justice_keyword_identification"
        }
        evaluator.dialogue_summaries = {}
        evaluator.context_patterns = {}

        evaluator.evaluate_dataset(
            input_file=input_file,
            output_file=output_file_mode1,
            delay=delay_between_requests,
            model=evaluation_model,
            use_cumulative_context=False,
            analyze_patterns=False,
            include_current_conversation=False
        )

        # Store results for overall summary
        mode1_result_key = f"eval_{base_name}_mode1_background_only_{evaluation_model.replace('-', '_')}.json"
        all_results[mode1_result_key] = {
            "total_questions": evaluator.results["total_questions"],
            "correct_answers": evaluator.results["correct_answers"],
            "accuracy": evaluator.results["accuracy"]
        }

        # MODE 2: Background + Current conversation (truncated)
        print(f"\\n--- Mode 2: 背景 + 当前对话(截断) ---")
        output_file_mode2 = os.path.join(output_dir, f"eval_{base_name}_mode2_with_current_{evaluation_model.replace('-', '_')}.json")

        # Reset results for new evaluation
        evaluator.results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "detailed_results": [],
            "task_type": "justice_keyword_identification"
        }
        evaluator.dialogue_summaries = {}
        evaluator.context_patterns = {}

        evaluator.evaluate_dataset(
            input_file=input_file,
            output_file=output_file_mode2,
            delay=delay_between_requests,
            model=evaluation_model,
            use_cumulative_context=False,
            analyze_patterns=False,
            include_current_conversation=True
        )

        # Store results for overall summary
        mode2_result_key = f"eval_{base_name}_mode2_with_current_{evaluation_model.replace('-', '_')}.json"
        all_results[mode2_result_key] = {
            "total_questions": evaluator.results["total_questions"],
            "correct_answers": evaluator.results["correct_answers"],
            "accuracy": evaluator.results["accuracy"]
        }



        # MODE 3: Background + Current conversation (truncated) + Previous 3 conversations (full text)
        print(f"\\n--- Mode 3: 背景 + 当前对话(截断) + 前3个对话(完整) ---")
        output_file_mode3 = os.path.join(output_dir, f"eval_{base_name}_mode3_with_previous_{evaluation_model.replace('-', '_')}.json")

        # Reset results for new evaluation
        evaluator.results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": 0.0,
            "detailed_results": [],
            "task_type": "justice_keyword_identification"
        }
        evaluator.dialogue_summaries = {}
        evaluator.context_patterns = {}

        evaluator.evaluate_dataset(
            input_file=input_file,
            output_file=output_file_mode3,
            delay=delay_between_requests,
            model=evaluation_model,
            use_cumulative_context=True,
            analyze_patterns=False,
            include_current_conversation=True
        )

        # Store results for overall summary
        mode3_result_key = f"eval_{base_name}_mode3_with_previous_{evaluation_model.replace('-', '_')}.json"
        all_results[mode3_result_key] = {
            "total_questions": evaluator.results["total_questions"],
            "correct_answers": evaluator.results["correct_answers"],
            "accuracy": evaluator.results["accuracy"]
        }

        print(f"{json_file}的评估完成 (生成了3个模式)")

    print(f"\\n{'='*60}")
    print("所有评估完成")
    print(f"{'='*60}")
    print(f"结果保存在: {output_dir}")
    print(f"评估的文件: {len(json_files)} 个文件 × 每个3个模式")

    # Generate overall accuracy summary
    save_overall_accuracy_summary(output_dir, evaluation_model, all_results)

if __name__ == "__main__":
    main()