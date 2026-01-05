from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def load_vector_store():
    if not os.path.exists("./chroma_db"):
        raise FileNotFoundError("Vector store not found. Run data_ingestion.py first.")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore

def generate_answer(query, context_docs):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("MODEL_NAME")
    )
    
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        error_msg = f"Answer generation failed: {str(e)}"
        return error_msg

def main():
    print("Starting RAG pipeline...")
    
    vectorstore = load_vector_store()
    print("Vector store loaded")
    
    query = "What are transformer architectures?"
    print(f"Query: {query}")
    
    results = vectorstore.similarity_search(query, k=5)
    print(f"Retrieved {len(results)} documents")
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown').split('/')[-1]
        print(f"\n{i}. {source}")
        print(f"   {doc.page_content[:100]}...")
    
    print("\n--- Generated Answer ---")
    answer = generate_answer(query, results)
    print(answer)

if __name__ == "__main__":
    main()
