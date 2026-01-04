ingestion pipeline using recursice char chunking

rag pipeline:
basic approach
vector search

improved approach
hybrid search(sparse + dense) -> rrf -> reranking

What it does:
- **Step 1**: Load vector database
- **Step 2**: Define test query ("What are transformer architectures?")
- **Step 3**: Retrieve 3 most similar documents using cosine similarity
- **Step 4**: Display retrieved documents (source + preview)
- **Step 5**: Generate and display final answer

## 🎯 Overall Script Purpose

Complete Simple RAG System:
1. Retrieval: Find 3 most semantically similar document chunks
2. Augmentation: Combine chunks into context for LLM
3. Generation: Use LLM to answer question based on retrieved context

## 📊 Execution Flow
Query → Vector Search → Top 3 Docs → Context Assembly → LLM → Answer


This is your baseline - simple cosine similarity retrieval with direct LLM
generation. 