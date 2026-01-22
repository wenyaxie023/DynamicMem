import os
import requests
from hipporag import HippoRAG

def check_service(url):
    try:
        if not url.endswith("/v1"):
            url = f"{url.rstrip('/')}/v1"
        response = requests.get(f"{url}/models")
        return response.status_code == 200
    except:
        return False

def main():
    # User's desired config (conceptually)
    # Reasoner: Qwen on Port 8000
    # Extractor: Llama on Port 8001
    
    reasoner_url = "http://localhost:8000/v1"
    extractor_url = "http://localhost:8001/v1"
    
    print("Checking vLLM services...")
    if not check_service(reasoner_url.replace('/v1', '')):
        print(f"Warning: Reasoner at {reasoner_url} is not responding yet.")
    else:
        print(f"Reasoner at {reasoner_url} is UP.")

    if not check_service(extractor_url.replace('/v1', '')):
        print(f"Warning: Extractor at {extractor_url} is not responding yet.")
    else:
        print(f"Extractor at {extractor_url} is UP.")

    print("\nInitializing HippoRAG...")
    # Note: HippoRAG 2 alpha source code shows __init__ takes args directly, not a config dict.
    # We will initialize it pointing to the Reasoner (Qwen) as the main LLM.
    # The separation of Reasoner/Extractor might require internal code changes or specific flags 
    # not visible in the top-level init. For now, we verify we can start it with Qwen.
    
    try:
        hrag = HippoRAG(
            llm_model_name="Qwen/Qwen2.5-14B-Instruct",
            llm_base_url=reasoner_url,
            embedding_model_name="Transformers/BAAI/bge-m3"
        )
        print("HippoRAG initialized successfully with Qwen 14B!")
        
        # --- Example Corpus ---
        print("\n--- Starting Example Indexing ---")
        corpus = [
            "Stanford University is located in California. It was founded by Leland Stanford.",
            "Leland Stanford was a tycoon and politician who served as the 8th Governor of California.",
            "Google was founded by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University."
        ]
        
        # Indexing (Builds Knowledge Graph)
        # This will use Qwen (Reasoner) to extract triples.
        hrag.index(corpus)
        print("Indexing completed.")
        
        # --- Example Retrieval & QA ---
        print("\n--- Starting Example Retrieval & QA ---")
        query = "Who founded the university in California where Google founders studied?"
        
        # 1. Retrieval
        # Returns a list of QuerySolution objects
        # We also pass gold_docs=None (default) so we get just the solutions.
        retrieval_results = hrag.retrieve(queries=[query], num_to_retrieve=1)
        
        for res in retrieval_results:
            print(f"\nQuery: {res.question}")
            print(f"Retrieved Docs: {res.docs}")
            print(f" Scores: {res.doc_scores}")

        # 2. QA
        # rag_qa takes the retrieval results (which can contain previously retrieved docs)
        # or raw queries (in which case it retrieves again).
        # We can pass the retrieval results directly to save time if we already ran retrieve.
        # But looking at rag_qa signature: rag_qa(queries: List[str|QuerySolution], ...)
        qa_results, _, _ = hrag.qa(queries=retrieval_results)
        
        for res in qa_results:
            print(f"\nAnswer: {res.answer}")

    except Exception as e:
        print(f"Error during HippoRAG execution: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nExample Run Completed.")

if __name__ == "__main__":
    main()
