from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os

load_dotenv()

class HybridRAGPipeline:
    def __init__(self, persist_directory="./chroma_db", config=None):
        self.config = config or {
            'dense_k': 8,     
            'sparse_k': 4,   
            'rerank_k': 5,  
            'temperature': 0.1
        }
        
        self.vectorstore = self._load_vectorstore(persist_directory)
        self.bm25_retriever = self._build_bm25_index()
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("MODEL_NAME"),
            temperature=self.config['temperature']
        )
        
    def _load_vectorstore(self, persist_directory):
        if not os.path.exists(persist_directory):
            raise FileNotFoundError("Vector store not found. Run data_ingestion.py first.")
        
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
    
    def _reciprocal_rank_fusion(self, dense_docs, sparse_docs, k=60):
        """Reciprocal Rank Fusion for combining dense and sparse retrieval results"""
        rrf_scores = {}
        
        for rank, doc in enumerate(dense_docs):
            doc_id = id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        for rank, doc in enumerate(sparse_docs):
            doc_id = id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        all_docs = dense_docs + sparse_docs
        unique_docs = []
        seen_ids = set()
        
        for doc in all_docs:
            doc_id = id(doc)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append((doc, rrf_scores[doc_id]))
        
        unique_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in unique_docs]
    
    def hybrid_retrieval(self, query):
        """Hybrid retrieval using Reciprocal Rank Fusion (RRF)"""
        dense_docs = self.vectorstore.similarity_search(query, k=self.config['dense_k'])
        
        self.bm25_retriever.k = self.config['sparse_k']
        sparse_docs = self.bm25_retriever.invoke(query)
        
        fused_docs = self._reciprocal_rank_fusion(dense_docs, sparse_docs)
        
        top_docs = fused_docs[:self.config['dense_k'] + self.config['sparse_k']]
        
        final_docs = self._rerank_documents(query, top_docs, self.config['rerank_k'])
        
        return final_docs
    
    def _rerank_documents(self, query, documents, top_k):
        if len(documents) <= top_k:
            return documents
            
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        reranked = [doc for doc, _ in scored_docs[:top_k]]
        
        return reranked
    
    def generate_answer(self, query, context_docs):
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            error_msg = f"Answer generation failed: {str(e)}"
            return error_msg
    
    def query(self, question):
        docs = self.hybrid_retrieval(question)
        
        answer = self.generate_answer(question, docs)
        
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
