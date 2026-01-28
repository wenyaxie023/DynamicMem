import igraph as ig
import sys
import json
from hashlib import md5
import argparse
import re

def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()

def inspect_graph(pickle_path, openie_json_path=None):
    try:
        g = ig.Graph.Read_Pickle(pickle_path)
        print(f"Graph: {pickle_path}")
        print(f"Nodes: {g.vcount()}")
        print(f"Edges: {g.ecount()}")
        
        triple_lookup = {}
        if openie_json_path:
            print(f"Loading OpenIE results from {openie_json_path} for relationship lookup...")
            with open(openie_json_path, 'r') as f:
                openie_data = json.load(f)
                for chunk in openie_data:
                    if not isinstance(chunk, dict): continue
                    for triple in chunk.get('extracted_triples', []):
                        if isinstance(triple, list) and len(triple) == 3:
                            # IMPORTANT: Match HippoRAG's internal processing order
                            p_head = text_processing(triple[0])
                            p_rel = text_processing(triple[1])
                            p_tail = text_processing(triple[2])
                            
                            head_hash = compute_mdhash_id(p_head, "entity-")
                            tail_hash = compute_mdhash_id(p_tail, "entity-")
                            key = (head_hash, tail_hash)
                            if key not in triple_lookup:
                                triple_lookup[key] = set()
                            triple_lookup[key].add(p_rel)

        # Sample edges
        print(f"\n--- Scanning for Semantic Edges (Recent={args.recent}) ---")
        
        edges_to_scan = g.es
        if args.recent:
            edges_to_scan = reversed(g.es)
            
        found_labeled = 0
        scanned_count = 0
        
        
        for e in edges_to_scan:
            # Optimization: Most semantic edges (Facts/Log links) have weight 1.0. 
            # Synonymy edges are usually < 1.0. Skip them to find semantic edges faster.
            if e['weight'] < 0.99:
                continue

            scanned_count += 1
            if scanned_count > 50000 and found_labeled == 0:
                 print("Scanned 50k candidate edges (weight ~ 1.0) without finding a semantic match. Stopping scan.")
                 break
                 
            source_name = g.vs[e.source]['name']
            target_name = g.vs[e.target]['name']
            
            # Check both directions just in case
            labels = triple_lookup.get((source_name, target_name), [])
            
            if labels:
                label_str = f" [{', '.join(labels)}]"
                source_content = g.vs[e.source]['content']
                target_content = g.vs[e.target]['content']
                # Clean up content for display
                print(f"Edge: {source_content} --{label_str}--> {target_content}")
                found_labeled += 1
            
            if found_labeled >= args.limit:
                break
            
                
    except Exception as e:
        print(f"Error inspecting graph: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_path")
    parser.add_argument("--openie_json", help="Path to openie_results_ner_gpt-5-mini.json")
    parser.add_argument("--recent", action="store_true", help="Scan from the end of the edge list")
    parser.add_argument("--limit", type=int, default=10, help="Number of labeled edges to find")
    args = parser.parse_args()
    
    inspect_graph(args.pickle_path, args.openie_json)
