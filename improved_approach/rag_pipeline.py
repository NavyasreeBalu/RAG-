from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os

load_dotenv()

class HybridRAGPipeline:
    def __init__(self, persist_directory="./chroma_db", config=None):
        self.config = config or {
            'dense_k': 5,
            'sparse_k': 5,
            'rerank_k': 3,
            'temperature': 0.3
        }
        
        self.vectorstore = self._load_vectorstore(persist_directory)
        self.bm25_retriever = self._build_bm25_index()
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
            temperature=self.config['temperature'],
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        

        
    def _load_vectorstore(self, persist_directory):
        if not os.path.exists(persist_directory):
            raise FileNotFoundError("Vector store not found. Run ingestion_pipeline.py first.")
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    def _build_bm25_index(self):
        collection = self.vectorstore._collection
        all_data = collection.get()
        
        from langchain_core.documents import Document
        documents = []
        for i, (doc_id, content, metadata) in enumerate(zip(all_data['ids'], all_data['documents'], all_data['metadatas'])):
            documents.append(Document(page_content=content, metadata=metadata or {}))
        
        return BM25Retriever.from_documents(documents)
    
    def hybrid_retrieval(self, query):
        dense_docs = self.vectorstore.similarity_search(query, k=self.config['dense_k'])
        
        self.bm25_retriever.k = self.config['sparse_k']
        sparse_docs = self.bm25_retriever.invoke(query)
        
        all_docs = dense_docs + sparse_docs
        
        final_docs = self._rerank_documents(query, all_docs, self.config['rerank_k'])
        
        return final_docs
    
    def _rerank_documents(self, query, documents, top_k):
        if len(documents) <= top_k:
            return documents
            
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def generate_answer(self, query, context_docs):
        return f"Answer generation skipped (quota limit). Retrieved {len(context_docs)} relevant documents."
    
    def query(self, question):
        docs = self.hybrid_retrieval(question)
        
        # Generate answer
        answer = self.generate_answer(question, docs)
        
        # Return result
        return {
            'question': question,
            'answer': answer,
            'sources': docs,
            'num_sources': len(docs)
        }
    



def main():
    print("Starting Hybrid RAG pipeline...")
    
    rag = HybridRAGPipeline()
    print("Pipeline initialized successfully")
    
    query = "What are transformer architectures?"
    print(f"\nQuery: {query}")
    
    result = rag.query(query)
    
    print(f"\nRetrieved {result['num_sources']} documents")
    
    for i, doc in enumerate(result['sources'], 1):
        source = doc.metadata.get('source', 'Unknown').split('/')[-1]
        print(f"\n{i}. {source}")
        print(f"   {doc.page_content[:100]}...")
    
    print(f"\n--- Generated Answer ---")
    print(result['answer'])

if __name__ == "__main__":
    main()
