# TCE (TCE) Health Checklist

## 1. TCE Health
- **Retrieval Latency**: 
  - Simple queries (1-3 keys): < 2s
  - Complex queries (10+ keys): 10s - 300s (Normal behavior for graph walks)
- **Process Management**:
  - Check for stuck processes: `ps aux | grep tce.py`
  - Deadlock Resolution: Kill old processes and restart with `--resume`.

## 2. Verification
- **Token Usage**: Run `python3 analyze_generation_tokens.py --results path/to/results.json` to verify token consumption.
- **Deep Investigation**: Use `deep_log_investigation.py` to trace specific logs or queries if issues arise.
