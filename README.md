# RAG System - Gen101 AI Internship Assignment

A hybrid Retrieval-Augmented Generation (RAG) system that combines dense semantic search, sparse BM25 retrieval, and cross-encoder reranking for improved document retrieval performance.

## 🎯 Project Overview

This project implements and compares two RAG approaches:
- **Baseline**: Simple cosine similarity search (k=3)
- **Improved**: Hybrid retrieval with dense + sparse + reranking

The improved approach achieved **28.6% precision improvement** with **100-200% gains on technical queries**.

## 📁 Project Structure

```
gen101/
├── src/                           # Source code
│   ├── data_ingestion.py          # Document processing pipeline
│   ├── baseline_retriever.py      # Simple RAG implementation
│   ├── hybrid_retriever.py        # Advanced hybrid RAG
│   ├── evaluator.py               # LLM-as-judge evaluation
│   └── test_queries.py            # Evaluation queries
├── outputs/                       # Generated results
├── research_papers/               # Input documents (10 LLM papers)
├── SETUP.md                       # Detailed setup instructions
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

For detailed setup instructions, see **[SETUP.md](SETUP.md)**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Add your GROQ_API_KEY

# 3. Run data ingestion
python src/data_ingestion.py

# 4. Run evaluation
python src/evaluator.py
```

## 🔧 Implementation Details

### Hybrid Retrieval Strategy
- **Dense Search**: Semantic similarity (k=8)
- **Sparse Search**: BM25 keyword matching (k=4)  
- **Cross-Encoder Reranking**: Final relevance scoring (k=5)
- **Safety Mechanism**: Preserves top semantic results

### Evaluation Framework
- **LLM-as-Judge**: ChatGroq with balanced scoring prompt
- **Metrics**: Precision@5, NDCG@10, Context Relevance
- **Robust Parsing**: Handles LLM response variations

## 📊 Performance Results

| Metric | Baseline | Improved | Improvement |
|--------|----------|----------|-------------|
| Precision@5 | 2.8 | 3.6 | +28.6% |
| Technical Queries | 1.5 | 4.0 | +166.7% |
| Overall NDCG@10 | 0.65 | 0.78 | +20.0% |

## 🛠 Technical Stack

- **Vector Store**: ChromaDB with sentence-transformers
- **Sparse Retrieval**: BM25 via rank-bm25
- **Reranking**: Cross-encoder models
- **LLM Evaluation**: ChatGroq (Llama-3.1-70b)
- **Framework**: LangChain for RAG pipeline

## 📋 Requirements

- Python 3.8+
- GROQ API key for evaluation
- 4GB+ RAM for vector embeddings
- ~2GB storage for models and data

## 🎯 Key Features

✅ **Hybrid Retrieval**: Combines multiple search strategies  
✅ **Safety Mechanisms**: Prevents cross-encoder regression  
✅ **Comprehensive Evaluation**: Multiple metrics with LLM judging  
✅ **Production Ready**: Class-based design, error handling  
✅ **Optimized Performance**: Efficient model loading and caching  

---

**For complete setup instructions and troubleshooting, see [SETUP.md](SETUP.md)**
