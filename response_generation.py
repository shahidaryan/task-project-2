from langchain.llms import OpenAI
from utils.vector_store import VectorStore

# Configuration
VECTOR_DB_PATH = "vector_store"
embedding_model = OpenAIEmbeddings()
llm = OpenAI(model="gpt-4")

# Initialize vector database
vector_store = VectorStore(VECTOR_DB_PATH)

def generate_response(query, top_k=5):
    print("Embedding the query...")
    query_embedding = embedding_model.embed(query)

    print("Retrieving relevant chunks...")
    results = vector_store.search(query_embedding, top_k)
    relevant_chunks = [result["text"] for result in results]

    print("Generating response...")
    context = "\n".join(relevant_chunks)
    prompt = f"Based on the following context, answer the query: {query}\n\nContext:\n{context}"
    response = llm(prompt)
    print(f"Response:\n{response}")

if __name__ == "__main__":
    query = input("Enter your query: ")
    generate_response(query)
