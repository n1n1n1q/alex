import os
from .wiki_dataset import WikiDataset

class WikiRetriever:
    def __init__(self, use_vector_db=True):
        self.use_vector_db = use_vector_db
        self.collection = None
        
        print("[RAG] Loading Wiki Dataset...")
        # Using full=False downloads 'wiki_samples' (smaller) for testing. 
        # Set full=True for the complete wiki.
        self.dataset = WikiDataset(full=False) 
        
        if self.use_vector_db:
            self._setup_vector_db()
            
    def _setup_vector_db(self):
        try:
            import chromadb
            # Using basic ChromaDB Client (in-memory/ephemeral for now)
            client = chromadb.Client()
            self.collection = client.get_or_create_collection("minecraft_knowledge")
            
            # If empty, populate index
            if self.collection.count() == 0:
                print("[RAG] Building Index (this may take a moment)...")
                ids = []
                documents = []
                metadatas = []
                
                # Index first 200 articles for performance testing
                # Remove the [:200] slice to index everything
                for i in range(min(len(self.dataset), 200)):
                    try:
                        item = self.dataset[i]
                        # Flatten 'texts' list into single string
                        full_text = " ".join([t['text'] for t in item.get('texts', [])])
                        
                        # Only index if text is substantial
                        if len(full_text) > 50:
                            ids.append(f"doc_{i}")
                            # Truncate to 1000 chars for context window management
                            documents.append(full_text[:1000])
                            metadatas.append({"source": "minedojo_wiki"})
                    except Exception as e:
                        continue
                
                if documents:
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
                print(f"[RAG] Indexed {len(documents)} articles.")
                
        except ImportError:
            print("[RAG] CRITICAL: 'chromadb' not found. RAG disabled.")
            self.use_vector_db = False

    def retrieve(self, query: str, k: int = 2) -> list[str]:
        """
        Search the wiki for the given query.
        """
        if self.use_vector_db and self.collection:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=k
                )
                # Chroma returns a list of lists (one per query)
                return results['documents'][0]
            except Exception as e:
                print(f"[RAG Error] {e}")
                return []
        return []