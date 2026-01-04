# RAG Evaluation Metrics

## Selected Metrics for RAG Comparison

### 1. Precision@5
- **Purpose**: Measures how many of the top 5 retrieved documents are actually relevant
- **Formula**: (# relevant docs in top 5) / 5
- **Why**: Most practical for RAG (typically use top 3-5 chunks)
- **Example**: If 3 out of 5 retrieved docs are relevant → Precision@5 = 0.6

### 2. NDCG@10 (Normalized Discounted Cumulative Gain)
- **Purpose**: Considers both relevance AND ranking position
- **Range**: 0-1 (1 = perfect ranking)
- **Why**: Industry standard for ranking quality, penalizes relevant docs ranked lower
- **Benefit**: Comprehensive ranking evaluation

### 3. Context Relevance Score
- **Purpose**: Measures how well retrieved chunks relate to the query
- **Method**: LLM-as-judge rating chunks 1-5 for query relevance
- **Why**: Most important for RAG quality, shows real-world impact
- **Implementation**: Use prompt: "Rate how relevant this text chunk is to answering the query on a scale of 1-5"

## Implementation Strategy
- Use these 3 metrics to compare baseline (vector search) vs improved (hybrid search + RRF + reranking)
- Provides quantitative comparison for final report
- Covers practical relevance, ranking quality, and real-world usefulness
