python -m eval.eval_state_abstraction \
  --benchmark generation/rag/results/001_user_001/prediction/state_abstraction_subset_checkpoint.json \
  --prediction generation/rag/results/001_user_001/prediction/state_abstraction_results.json \
  --output generation/rag/results/001_user_001/eval/state_abstraction_eval_llm_judge.json \
  --enable-llm-judge \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --llm-max-workers 1
