# HippoRAG System & Graph Health Checklist (Updated)

This document serves as a persistent reference for verifying the health and validity of the HippoRAG indexing process.

## 1. System Health & Performance
- **LLM Engine**: `gpt-5-mini` via Azure (Latency: ~10-20s, stability: High).
- **Embedding Engine**: `text-embedding-3-small` via Azure (Verified concurrent support).
- **Throughput**: ~50-60s per batch (batch_size=4) per user.
- **Concurrent Load**: Running **10 users parallel** (Total TPM ~130k-150k, within 200k limit).
- **Environment**: Multi-user tmux sessions (`hippo_001` to `hippo_010`).

## 2. Graph Data Quality (Deep Verification)
Verified against `gpt-5-mini` outputs for User 003 & 004:

### Information Extraction Quality:
- **Accuracy**: `gpt-5-mini` effectively parses nested JSON logs into semantic triples.
  - *Example*: Successfully extracted `["lichen", "observed_on", "granite"]` from a descriptive trail log.
- **Edge Semantic Meaningfulness**: **[HIGH]**. Extracted relationships are human-readable and capture logic, not just co-occurrence.
  - *Verification Item*: Check for logic like `[Program] -> takes_place_at -> [Location]` or `[Person] -> role -> [Title]`.
  - *Confirmed Sample*: `["Elena Rodriguez", "is Club Secretary of", "Classic City Rotary"]` (User 003).
- **Detail**: Captures fine-grained details (e.g., "Buster Investigated near lakeside creek") which are critical for precision retrieval.

### Node/Edge Attributes:
- **`chunk` Nodes**: Store complete raw JSON log entries (Verified for User 003/004).
- **`entity` Nodes**: Successfully store conceptual entities (e.g., "2022 excavation results").
- **`weight`**:
  - Triples/Direct Links: Weight >= 1.0 (Summed if repeated).
  - Synonymy Links: Weight range [0.8, 1.0) based on `bge-m3` or `text-embedding-3-small` similarity.

## 3. Persistent Verification Commands
Run these to check health anytime:
- **Progress**: `python3 check_progress.py`
- **Graph Stats**: `python3 inspect_graph.py <pickle_path>`
- **Error Scan**: `grep -ri "Error code" logs_stable_4x/`

## 4. Latest Global Statistics
- **Total Concurrent Users**: 10
- **Average Node Count**: ~3,000 per user (early stage)
- **Average Edge Count**: ~800,000 per user (due to synonymy density)
- **Status**: Stable execution.

---
*Comprehensive verification completed on 2026-01-27.*
