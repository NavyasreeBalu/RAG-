# Improved RAG Approach

## Overview
Enhanced RAG implementation with improved retrieval strategies for better context quality and answer accuracy.

## Improvements Over Baseline
- **Reranking**: Cross-encoder reranking for better relevance
- **Query Expansion**: Expand queries with related terms
- **Hybrid Search**: Combine semantic + keyword search
- **Better Context**: Improved document selection and fusion

## Components
- `improved_retrieval.py` - Enhanced retrieval with multiple strategies
- `rag_pipeline.py` - Complete improved RAG system

## Usage
```bash
python3 improved_retrieval.py  # Test improved retrieval
python3 rag_pipeline.py        # Run complete improved system
```

## Expected Improvements
- Higher relevance scores for retrieved documents
- Better context coherence for LLM generation
- More accurate answers to complex queries
- Reduced hallucination through better evidence
