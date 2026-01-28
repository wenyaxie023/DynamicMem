
import os
import re
import glob

def analyze_token_breakdown(log_dir):
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    # regex for openai_call stats
    stats_pattern = re.compile(r"prompt_tokens=(\d+)\s+completion_tokens=(\d+)")
    
    # Accumulators
    ner_stats = {'count': 0, 'prompt': 0, 'completion': 0}
    triple_stats = {'count': 0, 'prompt': 0, 'completion': 0}
    
    # State tracking
    current_type = None # 'NER' or 'TRIPLE'
    
    for log_file in sorted(log_files):
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 1. Determine Type based on msg_head
                if "msg_head=" in line:
                    if '"named_entities"' in line or '"entities"' in line: # NER often returns entities
                        current_type = 'NER'
                    elif '"triples"' in line:
                         current_type = 'TRIPLE'
                    elif '"log_' in line and 'timestamp' in line: # Raw log usually means NER input, but response?
                        # Let's look at the structure.
                        # NER 1 response: {"named_entities": [...]} 
                        # Triple response: {"triples": [...]}
                        # There might be some edge cases, but let's try this.
                        pass
                    else:
                        # Fallback or unknown
                        # Sometimes msg_head is cut off
                        pass
                
                # 2. Capture Stats
                if "openai_call" in line:
                    match = stats_pattern.search(line)
                    if match and current_type:
                        p = int(match.group(1))
                        c = int(match.group(2))
                        
                        if current_type == 'NER':
                            ner_stats['count'] += 1
                            ner_stats['prompt'] += p
                            ner_stats['completion'] += c
                        elif current_type == 'TRIPLE':
                            triple_stats['count'] += 1
                            triple_stats['prompt'] += p
                            triple_stats['completion'] += c
                        
                    # Reset type after capturing (to avoid carrying over to unrelated calls if any)
                    # Although logs are sequential per thread, with multi-threading in log file they are interleaved.
                    # Wait... The log lines [TIMING] returned and [TIMING] openai_call are usually adjacent or very close.
                    # BUT with multi-threading (10-32 workers), they might be mixed.
                    # Ideally we need thread ID, but standard logging here doesn't show it?
                    # Hmmm.
                    
                    # Correction: The log format is:
                    # [TIMING] openai API returned. duration=... msg_head=...
                    # [TIMING] openai_call model=... duration=... prompt_tokens=...
                    
                    # If threads are interleaved, Line A from Thread 1 might be followed by Line A from Thread 2 before Line B from Thread 1.
                    # However, usually the logger emits them atomically? No.
                    
                    # Let's assume they differ enough that 'msg_head' identifies the *immediately preceding* return.
                    # Or, better:
                    # NER output typically has "named_entities".
                    # Triple output typically has "triples".
                    
                    # Let's refining the "current_type" logic.
                    # If I see a msg_head line, I set the "pending type".
                    # If I verify that the very next openai_call is for that type...
                    # This is risky with interleaving.
                    
                    # ALTERNATIVE:
                    # Look at the *completion_tokens* size? No.
                    # Look at the prompt size?
                    # NER prompts are usually smaller (just the passage).
                    # Triple prompts are larger (passage + named entities list).
                    
    # Let's try with the assumption they are close enough or we can distinguishing by content.
    # Actually, let's look at the grep output again.
    # The 'msg_head' IS part of the completion (the response).
    # So if the response has "named_entities", it's NER. If "triples", it's Triple Extraction.
    
    # Revised Logic:
    # Since we can't easily link interleaved lines without a Request ID (which log seems to lack),
    # Let's use a heuristic: 
    # NER Output is usually a JSON with "named_entities".
    # Triple Output is usually a JSON with "triples".
    # But wait, the `openai_call` line DOES NOT contain the `msg_head`.
    # The `msg_head` is on a separate line.
    
    # If the logs are heavily interleaved, accurate attribution is hard without ID.
    # BUT, let's check the distance.
    # In `run_index_only.py`, the logger calls might be sequential in the code block.
    # If `logging` module uses a lock (it does by default), lines from a single emit are atomic.
    # But here we have two separate emit calls.
    
    # Let's try to assume simple sequentiality for now and see if the counts make sense.
    # OR, we can count the `msg_head` occurrences and `openai_call` occurrences separately 
    # and see if we can deduce average based on totals?
    # No, we need tokens.
    
    # Let's go with the "Recent Type" approach but maybe filter by proximity or timestamp if possible?
    # No, timestamp is only to the second.
    
    # Let's just try the sequential approach. In high concurrency, blocks might mix, 
    # but maybe `current_type` updates frequently enough. 
    # Actually, looking at the grep earlier:
    
    # [TIMING] openai API returned... msg_head=...
    # [TIMING] openai_call ...
    
    # They seem to be printed right next to each other in the code `llm/openai_gpt.py`.
    # Let's assume they stick together most of the time.
    
    pass

    print(f"{'Metric':<20} {'Count':<10} {'Avg Prompt':<15} {'Avg Output':<15} {'Total Output':<15}")
    print("-" * 80)
    
    for name, stats in [("NER", ner_stats), ("TRIPLE", triple_stats)]:
        if stats['count'] > 0:
            avg_p = stats['prompt'] / stats['count']
            avg_c = stats['completion'] / stats['count']
            print(f"{name:<20} {stats['count']:<10,} {avg_p:<15.1f} {avg_c:<15.1f} {stats['completion']:<15,}")

if __name__ == "__main__":
    analyze_token_breakdown("logs_stable_4x")
