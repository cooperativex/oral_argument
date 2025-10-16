# Oral Argument Comprehension Evaluation

This repository contains code and data for evaluating large language models on their ability to understand and reason about U.S. Supreme Court oral arguments.

## Overview

This research project evaluates LLMs across four distinct tasks that test different aspects of legal dialogue comprehension:

1. **Intention Analysis** - Multiple-choice questions about judicial motivations and reasoning patterns
2. **Next-Question Prediction** - Questions predicting the next statement in oral argument dialogue
3. **Advocate Keyword Identification** - Fill-in-the-blank questions testing recognition of key legal terms in advocate statements
4. **Justice Keyword Identification** - Fill-in-the-blank questions testing recognition of key legal terms in justice questions



## Dataset Structure 

```
oral_argument/
├── data/
│   ├── raw_data/                    # PDF transcripts from Oyez
│   ├── generate_advocate_questions/ # Advocate keyword questions (JSON)
│   ├── generate_justice_questions/  # Justice keyword questions (JSON)
│   ├── intention_questions/         # Judicial intention questions (JSON)
│   └── prediction_questions/        # Next-question prediction questions (JSON)

```


## Running Evaluations

Each evaluation script processes all JSON files in its respective input directory and generates results with three context modes:

- **Mode 1**: Background context only
- **Mode 2**: Background + current conversation (truncated before answer)
- **Mode 3**: Background + previous 3 conversations + current conversation

### Example: Advocate Keyword Evaluation

```bash
cd src/evaluation/Qwen2.5/advocate
python qwen2.5_70B_generate_new_2.py
```

The script will:
1. Load the Qwen2.5-72B model once
2. Process all files in `data/output_advocate_keywords/`
3. Generate 3 evaluation modes per file
4. Save results to `result_mode/generate_result/qwen2.5_72b_instruct/`
5. Create an overall accuracy summary

### Running Other Tasks

```bash
# Justice keyword evaluation
cd src/evaluation/Qwen2.5/justice
python qwen2.5_70B_generate_new_2.py

# Intention analysis
cd src/evaluation/Qwen2.5/intention
python qwen2.5_70b_eval_intention.py

# Prediction task
cd src/evaluation/Qwen2.5/prediction
python qwen2.5_70b_eval_predict.py
```

## Evaluation Methodology

### Answer Leakage Prevention

All evaluations implement strict security measures to prevent answer leakage:

- **Advocate/Justice Tasks**: Dialogue is truncated at the exact position where the original statement (containing the answer) begins
- **Intention/Prediction Tasks**: Dialogue is cut before the question that contains the answer
- Multiple validation layers verify answers don't appear in the context provided to the model


## Data Format

### Question Files

```json
{
  "source_file": "Case_Name.pdf",
  "background": "Case background context...",
  "conversations": [
    {
      "conversation_id": 1,
      "dialogue": "Full dialogue text...",
      "questions": [
        {
          "question": "Question text...",
          "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
          "answer": "A",
          "metadata": {
            "correct_phrase": "circuit split",
            "category": "case_law",
            "difficulty": "hard",
            "context_info": {...}
          }
        }
      ]
    }
  ]
}
```

### Result Files

```json
{
  "evaluation_timestamp": "2025-10-16T...",
  "total_questions": 100,
  "correct_answers": 75,
  "accuracy": 0.75,
  "task_type": "advocate_keyword_identification",
  "model_used": "qwen2.5-72b-instruct",
  "used_cumulative_context": true,
  "detailed_results": [...]
}
```

## Results

Results are saved in `result_mode/` with the following structure:
- Individual evaluation JSON files with detailed per-question results
- Overall accuracy summary text files comparing the three evaluation modes
- Breakdown by conversation and case


