from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

def load_vector_store():
    if not os.path.exists("./chroma_db"):
        raise FileNotFoundError("Vector store not found. Run ingestion_pipeline.py first.")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore

def retrieve_documents(vectorstore, query, k=3):
    results = vectorstore.similarity_search(query, k=k)
    return results

def generate_answer(query, context_docs):
    # Skip LLM generation due to quota limits  
    return f"Answer generation skipped (quota limit). Retrieved {len(context_docs)} relevant documents."

def main():
    print("Starting RAG pipeline...")
    
    vectorstore = load_vector_store()
    print("Vector store loaded")
    
    query = "What are transformer architectures?"
    print(f"Query: {query}")
    
    results = retrieve_documents(vectorstore, query, k=3)
    print(f"Retrieved {len(results)} documents")
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown').split('/')[-1]
        print(f"\n{i}. {source}")
        print(f"   {doc.page_content[:100]}...")
    
    print("\n--- Generated Answer ---")
    answer = generate_answer(query, results)
    print(answer)

main()
