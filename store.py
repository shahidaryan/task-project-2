import os
from langchain.vectorstores import FAISS

class VectorStore:
    def __init__(self, db_path):
        self.db_path = db_path
        if os.path.exists(db_path):
            self.db = FAISS.load_local(db_path)
        else:
            self.db = None

    def add(self, text, embedding, metadata=None):
        if self.db is None:
            self.db = FAISS.from_texts([text], [embedding], metadatas=[metadata])
        else:
            self.db.add_texts([text], [embedding], metadatas=[metadata])

    def search(self, query_embedding, top_k=5):
        return self.db.similarity_search_by_vector(query_embedding, top_k)

    def save(self):
        self.db.save_local(self.db_path)
