# Baseline RAG Approach

## Overview
Simple RAG implementation using similarity search for document retrieval.

## Components
- `ingestion_pipeline.py` - Load PDFs, chunk text, create vector database
- `rag_pipeline.py` - **Complete RAG pipeline** (retrieval + generation)

## Usage
1. **Setup vector database** (run once):
   ```bash
   python3 ingestion_pipeline.py
   ```

2. **Run RAG pipeline**:
   ```bash
   python3 rag_pipeline.py
   ```

## Approach
- **Retrieval**: Simple cosine similarity search (k=3)
- **Embeddings**: HuggingFace all-MiniLM-L6-v2
- **Generation**: Google Gemini 2.5 Flash
- **Vector Store**: Chroma DB

## Performance
- Retrieves relevant documents for transformer/LLM queries
- Generates accurate answers based on research paper context
- Baseline for comparison with improved approaches
