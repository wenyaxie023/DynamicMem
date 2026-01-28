
import os
import re
import glob
import pandas as pd


def investigate_log(log_file):
    print(f"Investigating: {log_file}")
    
    data = []
    
    # Regex to extract items
    # We look for lines with openai_call
    # And we look for lines with msg_head. 
    # Since they might be separated, we can't 100% link them without ID, 
    # but we can analyze the population of "msg_headers" vs "token_stats".
    
    msg_heads = []
    token_stats = []
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "msg_head=" in line:
                # Extract the content after msg_head=
                try:
                    head_content = line.split("msg_head=", 1)[1].strip()
                    msg_heads.append(head_content)
                except:
                    pass
            
            if "openai_call" in line:
                match = re.search(r"prompt_tokens=(\d+)\s+completion_tokens=(\d+)", line)
                if match:
                    token_stats.append({
                        'prompt': int(match.group(1)),
                        'completion': int(match.group(2))
                    })
    
    print(f"Found {len(msg_heads)} message headers.")
    print(f"Found {len(token_stats)} token stat entries.")
    
    # Analyze Message Headers
    ner_count = 0
    triple_count = 0
    unknown_count = 0
    
    for h in msg_heads:
        if '"named_entities"' in h or "'named_entities'" in h:
            ner_count += 1
        elif '"triples"' in h or "'triples'" in h:
            triple_count += 1
        else:
            unknown_count += 1
            # print sample of unknown
            # if unknown_count <= 5:
            #    print(f"Unknown head sample: {h[:100]}")

    print(f"\n--- Header Analysis ---")
    print(f"NER headers: {ner_count}")
    print(f"Triple headers: {triple_count}")
    print(f"Unknown headers: {unknown_count}")
    
    # Analyze Token Stats Distribution
    if not token_stats:
        print("No token stats found.")
        return

    df = pd.DataFrame(token_stats)
    
    print(f"\n--- Token Stats Analysis ---")
    print(df.describe())
    
    # Check for bimodality in Prompt Tokens (roughly 800 vs 1000?)
    # Simple histogram printing
    print("\nPrompt Token Dist (Bins of 100):")
    prompt_bins = [0, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 2000, 5000]
    out = pd.cut(df['prompt'], bins=prompt_bins).value_counts().sort_index()
    print(out)

    print("\nCompletion Token Dist (Bins of 500):")
    compl_bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 10000]
    out_c = pd.cut(df['completion'], bins=compl_bins).value_counts().sort_index()
    print(out_c)

if __name__ == "__main__":
    investigate_log("logs_stable_4x/001_user_001.log")
