# Langmem Baseline

## Note: QA is implemented while Dynamic State Prediction haven't yet.

## Output Layout

- QA:
  - `generation/Langmem/results/<user_id>/prediction/langmem_{args.mode}_results.json`

- Dynamic state prediction:
  - Not Implemented Yet 


## Input Data Requirements

- App_logs: 
  -`{args.input-root-dir}` or `.data/data_construction/generated_outputs/gemini_3_flash_preview"`

- QA:
  -`{args.qa-path}` or `generation/qa"`


## Mode Choice

- args.mode
  - `{hybrid},{triple},{profile},{episodic}`


## Run

- Python
  - `python generation/Langmem/langmem_full.py \
  --user-idx 001 \
  --mode hybrid \
  --input-root-dir /path/to/app_logs \
  --output-root-dir generation/Langmem/results \
  --qa-dir generation/qa \
  --top-k 5 \
  --embed-model openai:text-embedding-3-small \
  --llm-model gpt-4o-mini \
  --query-model gpt-4o-mini`

- Bash
  - 'bash generation/Langmem/run_langmem.sh'

