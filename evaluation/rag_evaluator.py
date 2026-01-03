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
        docs = retrieve_documents(self.baseline_vectorstore, query, k=3)
        retrieval_time = time.time() - start_time
        
        answer = generate_answer(query, docs)
        total_time = time.time() - start_time
        
        return {
            'docs': docs,
            'answer': answer,
            'retrieval_time': retrieval_time,
            'total_time': total_time,
            'num_sources': len(docs)
        }
    
    def evaluate_improved(self, query):
        return self.improved_rag.query(query)
    
    def compare_approaches(self):
        results = {}
        
        print(f"\nRunning evaluation on {len(TEST_QUERIES)} queries...")
        
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n[{i}/{len(TEST_QUERIES)}] Evaluating: {query[:50]}...")
            
            # Evaluate baseline
            baseline_result = self.evaluate_baseline(query)
            
            # Evaluate improved
            improved_result = self.evaluate_improved(query)
            
            # Calculate improvements
            speed_improvement = ((baseline_result['total_time'] - improved_result['total_time']) / baseline_result['total_time']) * 100
            
            results[query] = {
                'baseline': baseline_result,
                'improved': improved_result,
                'speed_improvement': speed_improvement
            }
            
            print(f"   Baseline: {baseline_result['total_time']:.2f}s | Improved: {improved_result['total_time']:.2f}s | Improvement: {speed_improvement:.1f}%")
        
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
