#!/bin/bash
echo "Starting User 002..."
python -u generation/run_batch_generation.py --user_id 002_user_002
echo "User 002 Done. Starting User 004..."
python -u generation/run_batch_generation.py --user_id 004_user_004
echo "All Done."
