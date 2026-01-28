
import igraph as ig
import json
import re
from hashlib import md5
import os
import sys

def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()

def verify(user_id):
    user_dir = f"outputs/{user_id}_large"
    graph_path = f"{user_dir}/gpt-5-mini_text-embedding-3-small/graph.pickle"
    openie_path = f"{user_dir}/openie_results_ner_gpt-5-mini.json"

    if not os.path.exists(graph_path):
        print(f"Graph not found: {graph_path}")
        return

    print(f"Loading Graph: {graph_path}...")
    try:
        g = ig.Graph.Read_Pickle(graph_path)
        print(f"Nodes: {g.vcount()}, Edges: {g.ecount()}")
    except Exception as e:
        print(f"❌ FAILED to load graph: {e}")
        return

    if not os.path.exists(openie_path):
        print(f"OpenIE file not found: {openie_path}")
        return

    with open(openie_path, 'r') as f:
        openie_data = json.load(f)

    checked_count = 0
    found_count = 0
    
    docs = openie_data.get('docs', [])
    # Check 20 triples from the middle to ensure past checkpoint logic works
    for chunk in docs[20:30]: 
        triples = chunk.get('extracted_triples', [])
        for triple in triples:
            if len(triple) != 3: continue
            
            p_head = text_processing(triple[0])
            p_rel = text_processing(triple[1])
            p_tail = text_processing(triple[2])
            
            head_id = compute_mdhash_id(p_head, "entity-")
            tail_id = compute_mdhash_id(p_tail, "entity-")
            
            checked_count += 1
            
            try:
                v_head = g.vs.find(name=head_id)
                v_tail = g.vs.find(name=tail_id)
                eid = g.get_eid(v_head.index, v_tail.index, error=False)
                if eid != -1:
                    found_count += 1
            except:
                pass

    if checked_count > 0:
        print(f"Summary: Found {found_count}/{checked_count} triples in graph ({found_count/checked_count*100:.1f}%)")
    else:
        print("Summary: No triples checked.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        verify(sys.argv[1])
    else:
        for uid in ["005_user_005", "006_user_006", "009_user_009"]:
            print(f"\n--- Checking {uid} ---")
            verify(uid)
