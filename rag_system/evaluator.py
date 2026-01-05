import os
import re
import numpy as np
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from baseline_retriever import load_vector_store
from hybrid_retriever import HybridRAGPipeline
from test_queries import TEST_QUERIES

load_dotenv()

class RAGEvaluator:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.0
        )
        
    def score_document(self, query, document_text):
        """LLM-as-judge relevance scoring (1-5)"""
        prompt = f"""Evaluate how well this document helps answer the query on a scale of 1-5:

Query: {query}

Document: {document_text[:1000]}

Scoring Guidelines:
5 = Directly and comprehensively answers the query
4 = Contains most information needed to answer the query  
3 = Provides useful relevant information
2 = Contains some relevant information but limited
1 = Little to no relevant information

Be truthful in your judgment.
Please respond with only the integer score (1-5):"""
        
        try:
            response = self.llm.invoke(prompt)
            score = int(response.content.strip())
            return max(1, min(5, score))
        except:
            return 1
    
    def precision_at_5(self, scores):
        if not scores: return 0.0
        top_5 = scores[:5]
        relevant = sum(1 for score in top_5 if score >= 3)
        return relevant / len(top_5)
        
    def ndcg_at_5(self, scores):
        def dcg(scores):
            return sum(score / np.log2(i + 2) for i, score in enumerate(scores))
        
        top_5 = scores[:5]
        if not top_5: return 0.0
        
        actual_dcg = dcg(top_5)
        ideal_dcg = dcg(sorted(top_5, reverse=True))
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
    
    def evaluate_system(self, name, retrieval_func, query):
        try:
            docs = retrieval_func(query)
            scores = [self.score_document(query, doc.page_content) for doc in docs]
            
            if name == "Baseline":
                from baseline_retriever import generate_answer
                answer = generate_answer(query, docs)
            else:
                hybrid_rag = HybridRAGPipeline()
                answer = hybrid_rag.generate_answer(query, docs)
            
            return {
                'precision': self.precision_at_5(scores),
                'ndcg': self.ndcg_at_5(scores),
                'scores': scores,
                'docs': docs,
                'answer': answer
            }
        except Exception as e:
            print(f"Error in {name}: {e}")
            return {'precision': 0.0, 'ndcg': 0.0, 'scores': [], 'docs': [], 'answer': 'Error generating answer'}
    
    def run_evaluation(self):
        """Compare baseline vs hybrid on all queries"""
        print("Loading systems...")
        vectorstore = load_vector_store()
        hybrid_rag = HybridRAGPipeline()
        
        def baseline_retrieve(query):
            return vectorstore.similarity_search(query, k=5)
        
        def hybrid_retrieve(query):
            return hybrid_rag.hybrid_retrieval(query)
        
        results = []
        
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"[{i}/{len(TEST_QUERIES)}] {query[:50]}...")
            
            baseline = self.evaluate_system("Baseline", baseline_retrieve, query)
            hybrid = self.evaluate_system("Hybrid", hybrid_retrieve, query)
            
            print(f"  Baseline P@5: {baseline['precision']:.2f}, NDCG@5: {baseline['ndcg']:.2f}")
            print(f"  Hybrid P@5: {hybrid['precision']:.2f}, NDCG@5: {hybrid['ndcg']:.2f}")
            
            results.append({
                'query': query,
                'baseline': baseline,
                'hybrid': hybrid
            })
        
        self.save_report(results)
    
    def save_report(self, results):
        """Save results to markdown"""
        b_p5 = np.mean([r['baseline']['precision'] for r in results])
        h_p5 = np.mean([r['hybrid']['precision'] for r in results])
        b_ndcg = np.mean([r['baseline']['ndcg'] for r in results])
        h_ndcg = np.mean([r['hybrid']['ndcg'] for r in results])
        
        p5_improvement = ((h_p5 - b_p5) / b_p5 * 100) if b_p5 > 0 else 0
        ndcg_improvement = ((h_ndcg - b_ndcg) / b_ndcg * 100) if b_ndcg > 0 else 0
        
        report = f"""# RAG Evaluation Report

## Executive Summary
**Baseline Precision@5:** {b_p5:.2f}
**Hybrid Precision@5:** {h_p5:.2f}
**P@5 Improvement:** {p5_improvement:+.1f}%

**Baseline NDCG@5:** {b_ndcg:.2f}
**Hybrid NDCG@5:** {h_ndcg:.2f}
**NDCG@5 Improvement:** {ndcg_improvement:+.1f}%

## Detailed Query Results
"""
        
        for result in results:
            report += f"""### Q: {result['query']}

**Metrics:**
- Baseline P@5: {result['baseline']['precision']:.2f}, NDCG@5: {result['baseline']['ndcg']:.2f}
- Hybrid P@5: {result['hybrid']['precision']:.2f}, NDCG@5: {result['hybrid']['ndcg']:.2f}

**Baseline Retrieved Docs:**
"""
            for i, doc in enumerate(result['baseline']['docs'][:3], 1):
                source = doc.metadata.get('source', 'Unknown').split('/')[-1]
                report += f"{i}. {source}: {doc.page_content[:100]}...\n"
            
            report += f"""
**Hybrid Retrieved Docs:**
"""
            for i, doc in enumerate(result['hybrid']['docs'][:3], 1):
                source = doc.metadata.get('source', 'Unknown').split('/')[-1]
                report += f"{i}. {source}: {doc.page_content[:100]}...\n"
            
            report += f"""
**Baseline Answer:**
{result['baseline']['answer'][:200]}...

**Hybrid Answer:**
{result['hybrid']['answer'][:200]}...

---

"""
        
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/evaluation_report.md", "w") as f:
            f.write(report)
        
        print(f"\nReport saved to outputs/evaluation_report.md")
        print(f"P@5: {b_p5:.2f} → {h_p5:.2f} ({p5_improvement:+.1f}%)")
        print(f"NDCG@5: {b_ndcg:.2f} → {h_ndcg:.2f} ({ndcg_improvement:+.1f}%)")

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.run_evaluation()
