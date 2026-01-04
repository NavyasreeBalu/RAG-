import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline_approach.rag_pipeline import load_vector_store, retrieve_documents, generate_answer
from improved_approach.rag_pipeline import HybridRAGPipeline
from test_queries import TEST_QUERIES

class RAGEvaluator:
    def __init__(self):
        print("Initializing RAG Evaluator...")
        
        # Load baseline approach
        print("Loading baseline approach...")
        self.baseline_vectorstore = load_vector_store()
        
        # Load improved approach
        print("Loading improved approach...")
        self.improved_rag = HybridRAGPipeline()
        
        print("Evaluator initialized successfully!")
    
    def evaluate_baseline(self, query):
        start_time = time.time()
        # Get documents with relevance scores
        docs_with_scores = self.baseline_vectorstore.similarity_search_with_score(query, k=3)
        docs = [doc for doc, score in docs_with_scores]
        scores = [score for doc, score in docs_with_scores]
        retrieval_time = time.time() - start_time
        
        answer = generate_answer(query, docs)
        total_time = time.time() - start_time
        
        return {
            'docs': docs,
            'answer': answer,
            'retrieval_time': retrieval_time,
            'total_time': total_time,
            'num_sources': len(docs),
            'relevance_scores': scores,
            'avg_relevance': sum(scores) / len(scores) if scores else 0
        }
    
    def evaluate_improved(self, query):
        start_time = time.time()
        result = self.improved_rag.query(query)
        total_time = time.time() - start_time
        
        # Get relevance scores for improved approach
        docs_with_scores = self.improved_rag.vectorstore.similarity_search_with_score(query, k=len(result['sources']))
        score_map = {doc.page_content: score for doc, score in docs_with_scores}
        
        # Map scores to retrieved documents
        relevance_scores = []
        for doc in result['sources']:
            # Find closest match in score_map
            best_score = min(score_map.values()) if score_map else 1.0
            for content, score in score_map.items():
                if doc.page_content[:100] in content or content[:100] in doc.page_content:
                    best_score = score
                    break
            relevance_scores.append(best_score)
        
        # Add timing and relevance info
        result['total_time'] = total_time
        result['retrieval_time'] = total_time
        result['relevance_scores'] = relevance_scores
        result['avg_relevance'] = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        return result
    
    def compare_approaches(self):
        results = {}
        
        print(f"\nRunning evaluation on {len(TEST_QUERIES)} queries...")
        
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n[{i}/{len(TEST_QUERIES)}] Evaluating: {query[:50]}...")
            
            # Evaluate baseline
            baseline_result = self.evaluate_baseline(query)
            
            # Evaluate improved
            improved_result = self.evaluate_improved(query)
            
            # Calculate speed difference
            speed_difference = improved_result['total_time'] - baseline_result['total_time']
            speed_improvement = ((baseline_result['total_time'] - improved_result['total_time']) / baseline_result['total_time']) * 100
            
            # Use LangChain relevance scores
            baseline_relevance = baseline_result['avg_relevance']
            improved_relevance = improved_result['avg_relevance']
            
            # Lower scores are better in similarity search (distance), so flip the improvement calculation
            relevance_improvement = ((baseline_relevance - improved_relevance) / baseline_relevance * 100) if baseline_relevance > 0 else 0
            
            baseline_sources = [doc.metadata.get('source', '').split('/')[-1] for doc in baseline_result['docs']]
            improved_sources = [doc.metadata.get('source', '').split('/')[-1] for doc in improved_result['sources']]
            
            results[query] = {
                'baseline': baseline_result,
                'improved': improved_result,
                'baseline_sources': baseline_sources,
                'improved_sources': improved_sources,
                'speed_difference': speed_difference,
                'speed_improvement': speed_improvement,
                'baseline_relevance': baseline_relevance,
                'improved_relevance': improved_relevance,
                'relevance_improvement': relevance_improvement
            }
            
            print(f"   Baseline: {baseline_result['total_time']:.2f}s (rel: {baseline_relevance:.3f}) | Improved: {improved_result['total_time']:.2f}s (rel: {improved_relevance:.3f}) | Relevance improvement: {relevance_improvement:.1f}%")
        
        return results
    
    def generate_report(self, results):
        report = "# RAG System Comparison Report\n\n"
        
        # Summary statistics
        avg_baseline_time = sum(r['baseline']['total_time'] for r in results.values()) / len(results)
        avg_improved_time = sum(r['improved']['total_time'] for r in results.values()) / len(results)
        avg_speed_improvement = sum(r['speed_improvement'] for r in results.values()) / len(results)
        
        report += "## Summary\n\n"
        report += f"- **Total queries evaluated**: {len(results)}\n"
        report += f"- **Average baseline time**: {avg_baseline_time:.2f}s\n"
        report += f"- **Average improved time**: {avg_improved_time:.2f}s\n"
        report += f"- **Average speed improvement**: {avg_speed_improvement:.1f}%\n\n"
        
        # Detailed comparison table
        report += "## Detailed Comparison\n\n"
        report += "| Query | Baseline Time | Improved Time | Speed Improvement | Baseline Sources | Improved Sources |\n"
        report += "|-------|---------------|---------------|-------------------|------------------|------------------|\n"
        
        for query, result in results.items():
            baseline = result['baseline']
            improved = result['improved']
            improvement = result['speed_improvement']
            
            # Truncate long queries for table
            short_query = query[:40] + "..." if len(query) > 40 else query
            
            report += f"| {short_query} | {baseline['total_time']:.2f}s | {improved['total_time']:.2f}s | +{improvement:.1f}% | {baseline['num_sources']} | {improved['num_sources']} |\n"
        
        # Retrieved documents comparison for first query
        report += "\n## Sample Retrieval Comparison\n\n"
        first_query = list(results.keys())[0]
        first_result = results[first_query]
        
        report += f"**Query**: {first_query}\n\n"
        
        report += "### Baseline Retrieved Documents:\n"
        for i, doc in enumerate(first_result['baseline']['docs'], 1):
            source = doc.metadata.get('source', 'Unknown').split('/')[-1]
            report += f"{i}. **{source}**\n"
            report += f"   {doc.page_content[:150]}...\n\n"
        
        report += "### Improved Retrieved Documents:\n"
        for i, doc in enumerate(first_result['improved']['sources'], 1):
            source = doc.metadata.get('source', 'Unknown').split('/')[-1]
            report += f"{i}. **{source}**\n"
            report += f"   {doc.page_content[:150]}...\n\n"
        
        # Save report
        with open('comparison_report.md', 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: comparison_report.md")
        return report

def main():
    evaluator = RAGEvaluator()
    results = evaluator.compare_approaches()
    report = evaluator.generate_report(results)
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Evaluated {len(results)} queries")
    print("Report generated: comparison_report.md")

if __name__ == "__main__":
    main()
