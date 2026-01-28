
import igraph as ig
import json
import re
from hashlib import md5
import os

def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()

def verify():
    graph_path = "outputs/004_user_004_large/gpt-5-mini_text-embedding-3-small/graph.pickle"
    openie_path = "outputs/004_user_004_large/openie_results_ner_gpt-5-mini.json"

    if not os.path.exists(graph_path):
        print(f"Graph not found: {graph_path}")
        return

    print(f"Loading Graph: {graph_path}...")
    g = ig.Graph.Read_Pickle(graph_path)
    print(f"Nodes: {g.vcount()}, Edges: {g.ecount()}")

    with open(openie_path, 'r') as f:
        openie_data = json.load(f)

    # Sample 5 chunks and check their triples
    checked_count = 0
    found_count = 0
    
    print("\n--- Spot Check: Mapping OpenIE Triples to Graph Edges ---")
    
    docs = openie_data.get('docs', [])
    for chunk in docs[:10]: # Check first 10 chunks
        triples = chunk.get('extracted_triples', [])
        for triple in triples:
            if len(triple) != 3: continue
            
            p_head = text_processing(triple[0])
            p_rel = text_processing(triple[1])
            p_tail = text_processing(triple[2])
            
            head_id = compute_mdhash_id(p_head, "entity-")
            tail_id = compute_mdhash_id(p_tail, "entity-")
            
            checked_count += 1
            
            # Check if these nodes exist
            try:
                v_head = g.vs.find(name=head_id)
                v_tail = g.vs.find(name=tail_id)
                
                # Check if edge exists
                eid = g.get_eid(v_head.index, v_tail.index, error=False)
                if eid != -1:
                    edge = g.es[eid]
                    print(f"✅ Found: [{triple[0]}] --({triple[1]})--> [{triple[2]}] (Weight: {edge['weight']})")
                    found_count += 1
                else:
                    print(f"❌ Missing Edge: [{triple[0]}] --({triple[1]})--> [{triple[2]}]")
            except ValueError:
                print(f"❌ Missing Node(s): [{triple[0]}] or [{triple[2]}]")

    print(f"\nSummary: Found {found_count}/{checked_count} triples in graph.")

if __name__ == "__main__":
    verify()
