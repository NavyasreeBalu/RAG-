from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
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

def dense_retriever(vectorstore, query, k=3):
    return vectorstore.similarity_search(query, k=k)

def sparse_retrieval(vectorstore, query, k=3):
    # Get proper Document objects with metadata
    all_docs = vectorstore.similarity_search("", k=1000)
    texts = [doc.page_content for doc in all_docs]
    
    bm25_retriever = BM25Retriever.from_texts(texts)
    bm25_retriever.k = k
    bm25_results = bm25_retriever.invoke(query)
    
    # Map BM25 results back to original Documents with metadata
    result_docs = []
    for bm25_doc in bm25_results:
        for orig_doc in all_docs:
            if orig_doc.page_content == bm25_doc.page_content:
                result_docs.append(orig_doc)
                break
    
    return result_docs[:k]

def rerank_documents(query, documents, top_k=3):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)
    
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, _ in scored_docs[:top_k]]

def hybrid_retrieval(vectorstore, query, k=3):
    dense_docs = dense_retriever(vectorstore, query, k)
    sparse_docs = sparse_retrieval(vectorstore, query, k)
    
    # Combine and deduplicate
    seen = set()
    combined = []
    for doc in dense_docs + sparse_docs:
        content = doc.page_content[:100]
        if content not in seen:
            seen.add(content)
            combined.append(doc)
    
    return rerank_documents(query, combined[:k*2], k)

def generate_answer(query, context_docs):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""You are a helpful AI assistant. Based on the research papers provided, answer the question clearly and concisely.

Context from research papers:
{context}

Question: {query}

Please provide a clear answer based on the context above:"""
    
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Generation error: {str(e)}"

def main():
    print("Starting Hybrid RAG pipeline...")
    
    vectorstore = load_vector_store()
    print("Vector store loaded")
    
    query = "What are transformer architectures?"
    print(f"Query: {query}")
    
    results = hybrid_retrieval(vectorstore, query, k=3)
    print(f"Retrieved {len(results)} documents using hybrid search")
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown').split('/')[-1]
        print(f"\n{i}. {source}")
        print(f"   {doc.page_content[:100]}...")
    
    print("\n--- Generated Answer ---")
    answer = generate_answer(query, results)
    print(answer)

if __name__ == "__main__":
    main()
