import sys
sys.path.append('baseline_approach')
sys.path.append('improved_approach')

from baseline_approach.rag_pipeline import load_vector_store as load_baseline, retrieve_documents
from improved_approach.rag_pipeline import load_vector_store as load_improved, hybrid_retrieval
from sentence_transformers import CrossEncoder

def score_relevance(query, documents):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)
    return scores.mean()

def evaluate_retrieval():
    # Test queries designed to show hybrid search advantages
    test_queries = [
        # --- Category 1: The "Acronym" Test (Tests Exact Keyword Match) ---
        # Baseline often fails because it looks for generic "Sequence" or "Token" concepts.
        # Improved (BM25) catches the exact hyphenated terms.
        "What is the difference between RAG-Sequence and RAG-Token?",

        # --- Category 2: The "Math/Symbol" Test (Tests Precision) ---
        # Baseline hates single letters like 'r' or 'k'. 
        # Improved finds the exact variable definition in the LoRA paper.
        "What is the specific rank 'r' used in LoRA?",

        # --- Category 3: The "Distractor" Test (The Hardest One) ---
        # Baseline sees "Attention" and pulls the 2017 Transformer paper (Wrong).
        # Improved Reranker sees "Flash" + "Memory" and pulls the 2022 FlashAttention paper (Right).
        "How does FlashAttention optimize memory access?",

        # --- Category 4: The "Specific Fact" Test ---
        # Baseline pulls generic "limitations" from GPT-3 or LLaMA.
        # Improved finds the specific "U-shaped" curve mentioned in 'Lost in the Middle'.
        "What is the 'U-shaped' performance curve in long context models?",

        # --- Category 5: The "Reasoning" Test ---
        # Baseline pulls the Intro of InstructGPT (vague).
        # Improved finds the specific 3-step bulleted list in the Methods section.
        "Explain the three steps of RLHF used in InstructGPT.",
    ]
    
    # Load both systems
    baseline_store = load_baseline()
    improved_store = load_improved()
    
    results = []
    
    for query in test_queries:
        print(f"\n=== Query: {query} ===")
        
        # Baseline retrieval
        baseline_docs = retrieve_documents(baseline_store, query, k=3)
        baseline_score = score_relevance(query, baseline_docs)
        
        # Improved retrieval  
        improved_docs = hybrid_retrieval(improved_store, query, k=3)
        improved_score = score_relevance(query, improved_docs)
        
        # Store results
        results.append({
            'query': query,
            'baseline_docs': baseline_docs,
            'improved_docs': improved_docs,
            'baseline_score': baseline_score,
            'improved_score': improved_score
        })
        
        # Print comparison
        print(f"Baseline: {len(baseline_docs)} docs, relevance: {baseline_score:.3f}")
        print(f"Improved: {len(improved_docs)} docs, relevance: {improved_score:.3f}")
        
    return results

def print_report(results):
    print("\n" + "="*80)
    print("EVALUATION REPORT")
    print("="*80)
    
    total_baseline = sum(r['baseline_score'] for r in results)
    total_improved = sum(r['improved_score'] for r in results)
    
    print(f"OVERALL RELEVANCE SCORES:")
    print(f"Baseline Average: {total_baseline/len(results):.3f}")
    print(f"Improved Average: {total_improved/len(results):.3f}")
    print(f"Improvement: {((total_improved - total_baseline)/total_baseline)*100:.1f}%")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Query: {result['query']}")
        print(f"   Baseline Score: {result['baseline_score']:.3f}")
        print(f"   Improved Score: {result['improved_score']:.3f}")
        print("-" * 60)
        
        print("BASELINE DOCUMENTS:")
        for j, doc in enumerate(result['baseline_docs'], 1):
            source = doc.metadata.get('source', 'Unknown').split('/')[-1]
            print(f"  {j}. {source}: {doc.page_content[:100]}...")
            
        print("\nIMPROVED DOCUMENTS:")
        for j, doc in enumerate(result['improved_docs'], 1):
            source = doc.metadata.get('source', 'Unknown').split('/')[-1]
            print(f"  {j}. {source}: {doc.page_content[:100]}...")

if __name__ == "__main__":
    results = evaluate_retrieval()
    print_report(results)
