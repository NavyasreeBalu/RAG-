# RAG - Improved Retrieval-Augmented Generation System

## Setup & Usage

See [SETUP.md](SETUP.md) for installation instructions and usage commands.

## Problem Statement

Modern AI assistants use Retrieval-Augmented Generation (RAG), but even with retrieval in place, generated responses may contain fabricated information, rely on incomplete documents, or combine unrelated content from multiple sources. This project improves retrieval logic to ensure LLMs receive reliable, contextually relevant evidence before generating responses.

## Technologies Used

- **LangChain**: Framework for building RAG applications and document processing
- **Vector Store**: ChromaDB with sentence-transformers embeddings
- **LLM**: ChatGroq (Llama-3.1) for answer generation and evaluation

## Architecture Overview

**Data Ingestion Pipeline:**
```
Research Papers (PDFs)
    ↓
Text Extraction
    ↓
Recursive Character Chunking
    ↓
Sentence-Transformer Embeddings
    ↓
ChromaDB Vector Store
```
*Both baseline and improved approaches use the same ingestion pipeline*

**Baseline Approach:**
```
Query Input
    ↓
Similarity Search
    ↓
Final Results
```

**Improved Approach:**
```
Query Input
    ↓
Dense Search ──┐
               ├── Reciprocal Rank Fusion
Sparse Search ─┘
    ↓
Cross-Encoder Reranking
    ↓
Final Results
```

## Solution: Hybrid Retrieval Strategy

Our approach combines four complementary retrieval methods:

### Stage 1: Dense Semantic Search
- Retrieves candidates based on contextual meaning
- Captures conceptual relationships and synonyms

### Stage 2: Sparse Search
- Retrieves candidates based on exact keyword matches
- Excels at finding specific technical terms, numbers, and proper nouns

### Stage 3: Reciprocal Rank Fusion (RRF)
- Applies RRF formula: score = 1/(k + rank + 1) for each retrieval method
- Deduplicates and creates unified candidate pool

### Stage 4: Cross-Encoder Reranking
- Reranks all candidates using query-document interaction modeling
- Returns final most relevant documents

## Results

**Key Achievement: 51.7% improvement in retrieval precision**

| Metric | Baseline | Hybrid | Improvement |
|--------|----------|---------|-------------|
| Precision@5 | 0.29 | 0.44 | +51.7% |
| NDCG@5 | 0.65 | 0.78 | +0.2 |

- **Precision@5**: Percentage of top 5 documents that are relevant (score ≥3)
- **NDCG@5**: Normalized ranking quality of top 5 results

See [results.md](results.md) for complete evaluation results and analysis.

This hybrid approach successfully reduces fabricated information, improves document relevance, and enhances reliability for enterprise RAG applications.
