# RAG System Approach

## Problem
Basic RAG systems use simple similarity search which often misses relevant documents or returns incomplete information. I wanted to build something better that actually finds the right content.

## My Solution

### Data Preparation
I use recursive character text splitting to break down 10 LLM research papers into 1500-character chunks with 300-character overlap. This gives good context while keeping chunks manageable.

### Baseline Approach (Simple RAG)
Just basic vector similarity search:
- Load the papers into a Chroma vector database
- For any query, find the 3 most similar chunks using cosine similarity
- Feed those chunks to the LLM to generate an answer

Pretty straightforward but often misses good content.

### Improved Approach (Hybrid RAG)
I built a hybrid system that combines multiple retrieval methods:

1. **Dense Retrieval**: Semantic search using embeddings (finds 6 candidates)
2. **Sparse Retrieval**: BM25 keyword search (finds 3 candidates)
3. **Deduplication**: Remove duplicate chunks using source + content matching
4. **Cross-Encoder Reranking**: Use a neural model to score and rank the final 5 documents
5. **Safety Mechanism**: Always keep the top semantic result to prevent the reranker from being too aggressive

## Why This Works Better

The hybrid approach catches documents that pure semantic search misses. For example:
- Semantic search is great for conceptual queries
- BM25 is better for specific technical terms or paper names
- The cross-encoder reranker understands query-document relevance better than simple similarity

## Results
My improved system shows:
- 28.6% better precision overall
- 100-200% improvements on technical queries
- Finds relevant papers that the baseline completely misses

The key insight is that different retrieval methods are good at different things, so combining them gives much better coverage of relevant content.

## Technical Stack
- **Vector DB**: Chroma with HuggingFace embeddings
- **BM25**: LangChain's BM25Retriever
- **Reranking**: Cross-encoder model (ms-marco-MiniLM-L-6-v2)
- **LLM**: Groq's Llama3-8B for answer generation
- **Evaluation**: LLM-as-judge scoring with Precision@5 and relevance metrics