#!/bin/bash
export OPENIE_MAX_WORKERS=6

# Create logs directory
mkdir -p logs_stable_4x

# Define user list
USERS="001_user_001 002_user_002 003_user_003 004_user_004 005_user_005 006_user_006 007_user_007 008_user_008 009_user_009 010_user_010"

for USER_TRIPLE in $USERS; do
  SESSION_NAME="hippo_${USER_TRIPLE:0:3}" # e.g., hippo_001
  DATA_PATH="../../user_data/${USER_TRIPLE}/app_log_large.json"
  SAVE_DIR="outputs/${USER_TRIPLE}_large"
  LOG_FILE="logs_stable_4x/${USER_TRIPLE}.log"

  echo "Launching $USER_TRIPLE in session $SESSION_NAME..."

  # Check if session exists, kill if so
  tmux has-session -t $SESSION_NAME 2>/dev/null
  if [ $? == 0 ]; then
    tmux kill-session -t $SESSION_NAME
  fi

  tmux new-session -d -s $SESSION_NAME "export OPENAI_TIMING=1; export OPENIE_MAX_WORKERS=6; ./.venv/bin/python -u run_index_only.py --data_path $DATA_PATH --save_dir $SAVE_DIR --model gpt-5-mini --batch_size 6 --embedding_model text-embedding-3-small --log_file $LOG_FILE"
  
  # Small delay to avoid thundering herd on API
  sleep 2
done

echo "All 10 users launched."
