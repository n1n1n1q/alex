import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alex.rag.retriever import WikiRetriever

def main():
    print("--- 1. Initializing Retriever ---")
    # This should trigger the download and indexing (first time only)
    retriever = WikiRetriever()
    
    if not retriever.use_vector_db:
        print("[Error] Vector DB is not active. Check imports.")
        return

    print("\n--- 2. Testing Search Queries ---")
    
    queries = [
        "zombie information and drops",
        "iron sword usage",
    ]
    
    for q in queries:
        print(f"\nQuery: '{q}'")
        results = retriever.retrieve(q, k=1)
        
        if results:
            print(f"Result: {results[0][:200]}...")
        else:
            print("Result: [NO MATCH FOUND]")

if __name__ == "__main__":
    main()