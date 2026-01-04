import sys
import os

# Clear any cached modules
if 'rag_pipeline' in sys.modules:
    del sys.modules['rag_pipeline']

# Import baseline functions
baseline_path = os.path.join(os.path.dirname(__file__), '..', 'baseline_approach')
sys.path.insert(0, baseline_path)
import rag_pipeline as baseline_rag
sys.path.remove(baseline_path)

# Import improved class  
improved_path = os.path.join(os.path.dirname(__file__), '..', 'improved_approach')
sys.path.insert(0, improved_path)

# Clear cache again
if 'rag_pipeline' in sys.modules:
    del sys.modules['rag_pipeline']
    
import rag_pipeline as improved_rag
sys.path.remove(improved_path)

# Import test queries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation_queries import TEST_QUERIES as test_queries
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import numpy as np
import time

load_dotenv()

class AdvancedRAGEvaluator:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("MODEL_NAME"),
            temperature=0.1
        )
        
    def get_relevance_score(self, query, document_text):
        """LLM-as-judge relevance scoring (1-5) with aggressive prompt"""
        prompt = f"""You are evaluating document relevance. Be STRICT and use the full 1-5 scale.

Query: {query}

Document text: {document_text[:300]}

Rate this document's relevance to answering the query:
1 = Completely irrelevant (mentions different topics)
2 = Barely relevant (tangentially related)
3 = Somewhat relevant (related but incomplete)
4 = Very relevant (directly addresses query)
5 = Perfectly relevant (completely answers query)

Look at the actual content and be critical. Most documents should NOT be a 3.

Rating (just the number): """
        
        try:
            response = self.llm.invoke(prompt)
            score = int(response.content.strip())
            return max(1, min(5, score))
        except:
            return 1  # Default to lowest score to avoid masking differences
    
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
        print("Initializing Advanced RAG Evaluator...")
        
        vectorstore = baseline_rag.load_vector_store()
        hybrid_rag = improved_rag.HybridRAGPipeline()
        
        def baseline_retrieve(query):
            return baseline_rag.retrieve_documents(vectorstore, query, k=5)  # Use module.function
        
        def improved_retrieve(query):
            result = hybrid_rag.hybrid_retrieval(query)
            return result[:5]  # Ensure max 5 documents
        
        results = []
        
        print(f"\nRunning advanced evaluation on {len(test_queries)} queries...\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"[{i}/{len(test_queries)}] Query: {query[:60]}...")
            
            baseline_result = self.evaluate_approach("Baseline", baseline_retrieve, query)
            improved_result = self.evaluate_approach("Improved", improved_retrieve, query)
            
            # Safe percentage calculations
            precision_improvement = ((improved_result['precision_5'] - baseline_result['precision_5']) / max(baseline_result['precision_5'], 0.001)) * 100 if baseline_result['precision_5'] > 0 else 0
            ndcg_improvement = ((improved_result['ndcg_10'] - baseline_result['ndcg_10']) / max(baseline_result['ndcg_10'], 0.001)) * 100 if baseline_result['ndcg_10'] > 0 else 0
            relevance_improvement = ((improved_result['context_relevance'] - baseline_result['context_relevance']) / max(baseline_result['context_relevance'], 0.001)) * 100 if baseline_result['context_relevance'] > 0 else 0
            
            print(f"   Baseline: P@5={baseline_result['precision_5']:.3f}, NDCG@10={baseline_result['ndcg_10']:.3f}, Rel={baseline_result['context_relevance']:.3f}")
            print(f"   Improved: P@5={improved_result['precision_5']:.3f}, NDCG@10={improved_result['ndcg_10']:.3f}, Rel={improved_result['context_relevance']:.3f}")
            print(f"   Improvements: P@5={precision_improvement:+.1f}%, NDCG@10={ndcg_improvement:+.1f}%, Rel={relevance_improvement:+.1f}%")
            
            # Debug: Show actual scores and documents
            print(f"   Baseline scores: {baseline_result['relevance_scores'][:5]}")
            print(f"   Improved scores: {improved_result['relevance_scores'][:5]}")
            print(f"   Baseline docs: {[doc.metadata.get('source', '').split('/')[-1] for doc in baseline_result['documents'][:3]]}")
            print(f"   Improved docs: {[doc.metadata.get('source', '').split('/')[-1] for doc in improved_result['documents'][:3]]}")
            print(f"   Baseline content: {[doc.page_content[:50] for doc in baseline_result['documents'][:2]]}")
            print(f"   Improved content: {[doc.page_content[:50] for doc in improved_result['documents'][:2]]}\n")
            
            results.append({
                'query': query,
                'baseline': baseline_result,
                'improved': improved_result
            })
        
        self.generate_advanced_report(results)
        return results
    
    def generate_advanced_report(self, results):
        """Generate detailed evaluation report with document comparison"""
        report = "# Advanced RAG Evaluation Report\n\n"
        
        # Calculate averages
        avg_baseline_p5 = np.mean([r['baseline']['precision_5'] for r in results])
        avg_improved_p5 = np.mean([r['improved']['precision_5'] for r in results])
        avg_baseline_ndcg = np.mean([r['baseline']['ndcg_10'] for r in results])
        avg_improved_ndcg = np.mean([r['improved']['ndcg_10'] for r in results])
        avg_baseline_rel = np.mean([r['baseline']['context_relevance'] for r in results])
        avg_improved_rel = np.mean([r['improved']['context_relevance'] for r in results])
        
        report += f"## Summary\n\n"
        report += f"| Metric | Baseline | Improved | Improvement |\n"
        report += f"|--------|----------|----------|-------------|\n"
        report += f"| Precision@5 | {avg_baseline_p5:.3f} | {avg_improved_p5:.3f} | {((avg_improved_p5-avg_baseline_p5)/max(avg_baseline_p5, 0.001))*100:+.1f}% |\n"
        report += f"| NDCG@10 | {avg_baseline_ndcg:.3f} | {avg_improved_ndcg:.3f} | {((avg_improved_ndcg-avg_baseline_ndcg)/max(avg_baseline_ndcg, 0.001))*100:+.1f}% |\n"
        report += f"| Context Relevance | {avg_baseline_rel:.3f} | {avg_improved_rel:.3f} | {((avg_improved_rel-avg_baseline_rel)/max(avg_baseline_rel, 0.001))*100:+.1f}% |\n\n"
        
        # Document comparison for first query (addresses problem statement requirement)
        if results:
            first_result = results[0]
            report += "## Sample Document Comparison (Before vs After)\n\n"
            report += f"**Query**: {first_result['query']}\n\n"
            
            report += "### Baseline Retrieved Documents:\n"
            for i, doc in enumerate(first_result['baseline']['documents'][:3], 1):
                source = doc.metadata.get('source', 'Unknown').split('/')[-1]
                report += f"{i}. **{source}**\n   {doc.page_content[:100]}...\n\n"
            
            report += "### Improved Retrieved Documents:\n"
            for i, doc in enumerate(first_result['improved']['documents'][:3], 1):
                source = doc.metadata.get('source', 'Unknown').split('/')[-1]
                report += f"{i}. **{source}**\n   {doc.page_content[:100]}...\n\n"
        
        with open("advanced_evaluation_report.md", "w") as f:
            f.write(report)
        
        print(f"Advanced report saved to: advanced_evaluation_report.md")

if __name__ == "__main__":
    evaluator = AdvancedRAGEvaluator()
    evaluator.run_evaluation()
