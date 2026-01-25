
import os
import argparse
import json
from hipporag import HippoRAG

def test_qa(output_dir, reasoner_port):
    print(f"[*] Initializing HippoRAG from {output_dir}...")
    
    # Initialize with the directory containing the graph
    # Note: We must match the config used during indexing
    hipporag = HippoRAG(
        save_dir=output_dir,
        llm_model_name="Qwen/Qwen2.5-14B-Instruct",
        llm_base_url=f"http://localhost:{reasoner_port}/v1",
        embedding_model_name="Transformers/BAAI/bge-m3"
    )
    
    # Questions based on app_log_small.json content
    queries = [
        "What is the total balance in my Chase accounts?",
        "Where did I go hiking on October 1st?",
        "What message did I send to the Family Group on WhatsApp?"
    ]
    
    print("\n[*] Starting QA Test...")
    
    # 1. Retrieval Only Test
    print("\n--- Retrieval Check ---")
    retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=2)
    for i, res in enumerate(retrieval_results):
        print(f"\nQ: {res.question}")
        print("Retrieved Docs:")
        for doc in res.docs:
            # Truncate doc for display
            print(f" - {doc[:150]}...")

    # 2. End-to-End QA
    print("\n--- RAG QA Check ---")
    qa_results, _, _ = hipporag.rag_qa(queries=queries)
    
    for res in qa_results:
        print(f"\nQ: {res.question}")
        print(f"A: {res.answer}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reasoner_port", type=int, default=8000)
    args = parser.parse_args()
    
    test_qa(args.output_dir, args.reasoner_port)
