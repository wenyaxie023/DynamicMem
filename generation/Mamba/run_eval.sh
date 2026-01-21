#!/bin/bash
# pip install -r requirements.txt # user might need to install this

echo "Running Mamba evaluation..."
python3 eval_mamba.py --limit 5 --output_path mamba_results_sample.json
echo "Done."
