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
        
        # Load heavy resources once
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
        # More efficient: get all documents directly
        collection = self.vectorstore._collection
        all_data = collection.get()
        
        # Convert to Document objects with metadata
        from langchain.schema import Document
        documents = []
        for i, (doc_id, content, metadata) in enumerate(zip(all_data['ids'], all_data['documents'], all_data['metadatas'])):
            documents.append(Document(page_content=content, metadata=metadata or {}))
        
        return BM25Retriever.from_documents(documents)
    
    def _rrf_fusion(self, dense_docs, sparse_docs, k=60):
        # Reciprocal Rank Fusion
        doc_scores = {}
        all_docs = dense_docs + sparse_docs
        
        # Score dense results
        for rank, doc in enumerate(dense_docs):
            doc_scores[doc] = doc_scores.get(doc, 0) + 1 / (k + rank + 1)
        
        # Score sparse results  
        for rank, doc in enumerate(sparse_docs):
            doc_scores[doc] = doc_scores.get(doc, 0) + 1 / (k + rank + 1)
        
        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in sorted_docs[:self.config['rerank_k']*2]]
    
    def _dense_retrieval(self, query, k):
        return self.vectorstore.similarity_search(query, k=k)
    
    def _sparse_retrieval(self, query, k):
        self.bm25_retriever.k = k
        return self.bm25_retriever.invoke(query)
    
    def _rerank_documents(self, query, documents, top_k):
        if len(documents) <= top_k:
            return documents
            
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def hybrid_retrieval(self, query):
        # Retrieve from both methods
        dense_docs = self._dense_retrieval(query, self.config['dense_k'])
        sparse_docs = self._sparse_retrieval(query, self.config['sparse_k'])
        
        # RRF fusion
        fused_docs = self._rrf_fusion(dense_docs, sparse_docs)
        
        # Rerank
        final_docs = self._rerank_documents(query, fused_docs, self.config['rerank_k'])
        
        return final_docs
    
    def generate_answer(self, query, context_docs):
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = f"""Based on the research papers provided, answer the question clearly and concisely.

Context from research papers:
{context}

Question: {query}

Please provide a clear answer based on the context above:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Generation error: {str(e)}"
    
    def query(self, question):
        # Retrieve documents
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
    
    def evaluate_retrieval(self, test_queries):
        results = {}
        for query in test_queries:
            start_time = time.time()
            docs = self.hybrid_retrieval(query)
            
            results[query] = {
                'docs': docs,
                'latency': time.time() - start_time,
                'num_docs': len(docs),
                'sources': [doc.metadata.get('source', 'Unknown') for doc in docs]
            }
        return results
    


def main():
    print("Starting Hybrid RAG pipeline...")
    
    # Initialize pipeline
    rag = HybridRAGPipeline()
    print("Pipeline initialized successfully")
    
    # Test query
    query = "What are transformer architectures?"
    print(f"\nQuery: {query}")
    
    result = rag.query(query)
    
    print(f"\nRetrieved {result['num_sources']} documents in {result['retrieval_time']:.3f}s")
    
    for i, doc in enumerate(result['sources'], 1):
        source = doc.metadata.get('source', 'Unknown').split('/')[-1]
        print(f"\n{i}. {source}")
        print(f"   {doc.page_content[:100]}...")
    
    print(f"\n--- Generated Answer ---")
    print(result['answer'])

if __name__ == "__main__":
    main()
