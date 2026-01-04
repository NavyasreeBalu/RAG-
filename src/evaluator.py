import sys
import os
import numpy as np
import time
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from baseline_retriever import load_vector_store, generate_answer
from hybrid_retriever import HybridRAGPipeline
from test_queries import TEST_QUERIES

load_dotenv()

class RAG_Evaluator:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("MODEL_NAME"),
            temperature=0.1
        )
        
    def get_relevance_score(self, query, document_text):
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

Please respond with only the integer score (1-5):"""
        
        try:
            response = self.llm.invoke(prompt)
            score = int(response.content.strip())
            return max(1, min(5, score))
        except:
            return 1
    
    def calculate_precision_at_k(self, relevance_scores, k=5):
        """Calculate Precision@K"""
        actual_k = min(k, len(relevance_scores))  # Use actual number available
        top_k_scores = relevance_scores[:actual_k]
        relevant_count = sum(1 for score in top_k_scores if score >= 3)
        return relevant_count / actual_k  # Divide by actual k, not fixed k
    
    def calculate_ndcg_at_k(self, relevance_scores, k=10):
        """Calculate NDCG@K"""
        def dcg(scores):
            return sum(score / np.log2(i + 2) for i, score in enumerate(scores))
        
        actual_dcg = dcg(relevance_scores[:k])
        ideal_scores = sorted(relevance_scores[:k], reverse=True)
        ideal_dcg = dcg(ideal_scores)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
    
    def calculate_context_relevance(self, relevance_scores):
        """Average relevance score"""
        return np.mean(relevance_scores)
    
    def evaluate_approach(self, approach_name, retrieval_func, query):
        """Evaluate single approach with all metrics"""
        print(f"  Evaluating {approach_name}...")
        
        start_time = time.time()
        documents = retrieval_func(query)
        retrieval_time = time.time() - start_time
        
        relevance_scores = []
        for doc in documents:
            score = self.get_relevance_score(query, doc.page_content)
            relevance_scores.append(score)
        
        precision_5 = self.calculate_precision_at_k(relevance_scores, k=5)
        ndcg_10 = self.calculate_ndcg_at_k(relevance_scores, k=10)
        context_relevance = self.calculate_context_relevance(relevance_scores)
        
        return {
            'time': retrieval_time,
            'precision_5': precision_5,
            'ndcg_10': ndcg_10,
            'context_relevance': context_relevance,
            'documents': documents,
            'relevance_scores': relevance_scores
        }
    
    def run_evaluation(self):
        """Run complete evaluation"""
        print("Initializing RAG Evaluator...")
        
        vectorstore = load_vector_store()
        hybrid_rag = HybridRAGPipeline()
        
        def baseline_retrieve(query):
            return vectorstore.similarity_search(query, k=5)
        
        def improved_retrieve(query):
            result = hybrid_rag.hybrid_retrieval(query)
            return result[:5]  # Ensure max 5 documents
        
        results = []
        
        print(f"\nRunning evaluation on {len(TEST_QUERIES)} queries...\n")
        
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"[{i}/{len(TEST_QUERIES)}] {query[:60]}...")
            
            baseline_result = self.evaluate_approach("Baseline", baseline_retrieve, query)
            improved_result = self.evaluate_approach("Improved", improved_retrieve, query)
            
            # Safe percentage calculations
            precision_improvement = ((improved_result['precision_5'] - baseline_result['precision_5']) / max(baseline_result['precision_5'], 0.001)) * 100 if baseline_result['precision_5'] > 0 else 0
            ndcg_improvement = ((improved_result['ndcg_10'] - baseline_result['ndcg_10']) / max(baseline_result['ndcg_10'], 0.001)) * 100 if baseline_result['ndcg_10'] > 0 else 0
            relevance_improvement = ((improved_result['context_relevance'] - baseline_result['context_relevance']) / max(baseline_result['context_relevance'], 0.001)) * 100 if baseline_result['context_relevance'] > 0 else 0
            
            print(f"   Baseline: P@5={baseline_result['precision_5']:.3f}, Rel={baseline_result['context_relevance']:.3f}")
            print(f"   Improved: P@5={improved_result['precision_5']:.3f}, Rel={improved_result['context_relevance']:.3f}")
            print(f"   Change:   P@5={precision_improvement:+.1f}%, Rel={relevance_improvement:+.1f}%")
            print(f"   Sources:  {[doc.metadata.get('source', '').split('/')[-1][:25] for doc in improved_result['documents'][:3]]}\n")
            
            results.append({
                'query': query,
                'baseline': baseline_result,
                'improved': improved_result
            })
        
        self.generate_evaluation_report(results)
        return results
    
    def generate_evaluation_report(self, results):
        """Generate detailed evaluation report with visualizations"""
        report = "# RAG System Evaluation Report\n\n"
        
        # Calculate averages
        avg_baseline_p5 = np.mean([r['baseline']['precision_5'] for r in results])
        avg_improved_p5 = np.mean([r['improved']['precision_5'] for r in results])
        avg_baseline_rel = np.mean([r['baseline']['context_relevance'] for r in results])
        avg_improved_rel = np.mean([r['improved']['context_relevance'] for r in results])
        
        # Helper functions for visualization
        def get_bar(value, max_val=1.0, width=20):
            filled = int((value / max_val) * width)
            return '█' * filled + '░' * (width - filled)
        
        def get_arrow(improvement):
            if improvement > 50: return '↗↗↗'
            elif improvement > 10: return '↗↗'
            elif improvement > 0: return '↗ '
            elif improvement < -10: return '↘↘'
            elif improvement < 0: return '↘ '
            else: return '→ '
        
        report += f"## Performance Summary\n\n"
        report += f"| Metric | Baseline | Improved | Change |\n"
        report += f"|--------|----------|----------|--------|\n"
        report += f"| Precision@5 | {avg_baseline_p5:.3f} | {avg_improved_p5:.3f} | {((avg_improved_p5-avg_baseline_p5)/max(avg_baseline_p5, 0.001))*100:+.1f}% |\n"
        report += f"| Context Relevance | {avg_baseline_rel:.3f} | {avg_improved_rel:.3f} | {((avg_improved_rel-avg_baseline_rel)/max(avg_baseline_rel, 0.001))*100:+.1f}% |\n\n"
        
        report += f"## Query Performance Visualization\n\n"
        report += f"```\n"
        
        for i, result in enumerate(results, 1):
            query = result['query'][:50] + "..." if len(result['query']) > 50 else result['query']
            baseline_p5 = result['baseline']['precision_5']
            improved_p5 = result['improved']['precision_5']
            improvement = ((improved_p5 - baseline_p5) / max(baseline_p5, 0.001)) * 100 if baseline_p5 > 0 else 0
            
            report += f"[{i}] {query}\n"
            report += f"    Baseline: P@5={baseline_p5:.3f} {get_bar(baseline_p5)}\n"
            report += f"    Improved: P@5={improved_p5:.3f} {get_bar(improved_p5)} {get_arrow(improvement)} {improvement:+.1f}%\n\n"
        
        report += f"```\n\n"
        
        # Document comparison for first query
        if results:
            first_result = results[0]
            report += "## Sample Document Comparison\n\n"
            report += f"**Query**: {first_result['query']}\n\n"
            
            report += "### Baseline Retrieved Documents:\n"
            for i, doc in enumerate(first_result['baseline']['documents'][:3], 1):
                source = doc.metadata.get('source', 'Unknown').split('/')[-1]
                report += f"{i}. **{source}**\n   {doc.page_content[:100]}...\n\n"
            
            report += "### Improved Retrieved Documents:\n"
            for i, doc in enumerate(first_result['improved']['documents'][:3], 1):
                source = doc.metadata.get('source', 'Unknown').split('/')[-1]
                report += f"{i}. **{source}**\n   {doc.page_content[:100]}...\n\n"
        
        with open("outputs/evaluation_report.md", "w") as f:
            f.write(report)
        
        print(f"Report with visualizations saved to: outputs/evaluation_report.md")

if __name__ == "__main__":
    evaluator = RAG_Evaluator()
    evaluator.run_evaluation()
