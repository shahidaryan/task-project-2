from utils.vector_store import VectorStore
from langchain.embeddings import OpenAIEmbeddings

# Configuration
VECTOR_DB_PATH = "vector_store"
embedding_model = OpenAIEmbeddings()

# Initialize vector database
vector_store = VectorStore(VECTOR_DB_PATH)

def retrieve_chunks(query, top_k=5):
    print("Embedding the query...")
    query_embedding = embedding_model.embed(query)

    print("Retrieving relevant chunks...")
    results = vector_store.search(query_embedding, top_k)

    for result in results:
        print(f"Chunk: {result['text']}\nSource: {result['metadata']['url']}\n")
    return results

if __name__ == "__main__":
    query = input("Enter your query: ")
    retrieve_chunks(query)
