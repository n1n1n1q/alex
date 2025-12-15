import os
from .wiki_dataset import WikiDataset

class WikiRetriever:
    def __init__(self, use_vector_db=True):
        self.use_vector_db = use_vector_db
        self.collection = None
        
        print("[RAG] Loading Wiki Dataset...")

        repo_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )

        repo_root = "/datasets/maksymz/alex/alex/rag"
        # ...existing code...
        self.dataset = WikiDataset(
            download_dir=os.path.join(repo_root, "data"),
            full=True
        ) 
        
        if self.use_vector_db:
            self._setup_vector_db()
            
    def _setup_vector_db(self):
        try:
            import chromadb
            # ...existing code...
            client = chromadb.Client()
            self.collection = client.get_or_create_collection("minecraft_knowledge")
            
            # ...existing code...
            if self.collection.count() == 0:
                print("[RAG] Building Index (this may take a moment)...")
                ids = []
                documents = []
                metadatas = []
                
                # ...existing code...
                for i in range(min(len(self.dataset), 200)):
                    try:
                        item = self.dataset[i]
                        # ...existing code...
                        full_text = " ".join([t['text'] for t in item.get('texts', [])])
                        
                        # ...existing code...
                        if len(full_text) > 50:
                            ids.append(f"doc_{i}")
                            # ...existing code...
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
                # ...existing code...
                return results['documents'][0]
            except Exception as e:
                print(f"[RAG Error] {e}")
                return []
        return []