# RAG System Setup Guide

## Prerequisites
- Python 3.8+
- ~2GB disk space for models and vector database
- GROQ API key (free tier available at https://console.groq.com)

## Installation

### 1. Setup Environment
```bash
cd gen101
python3 -m venv venv
source venv/bin/activate 
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create `.env` file in project root:
```bash
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
echo "MODEL_NAME=llama-3.1-8b-instant" >> .env
```

## Usage

### 4. Build Knowledge Base 
```bash
python rag_system/data_ingestion.py
```
This processes the 10 research papers and creates the vector database in `chroma_db/`.

### 5. Test Individual Approaches

**Baseline RAG:**
```bash
python rag_system/baseline_retriever.py
```

**Hybrid RAG:**
```bash
python rag_system/hybrid_retriever.py
```

### 6. Run Complete Evaluation
```bash
python rag_system/evaluator.py
```
This compares both approaches using 21 test queries and generates:
- Console output with real-time results
- `outputs/evaluation_report.md` with detailed comparison
- Shows retrieved documents and generated answers for each query

### 7. View Results
```bash
cat outputs/evaluation_report.md
```

## Project Structure
```
gen101/
├── rag_system/
│   ├── data_ingestion.py      # Document processing
│   ├── baseline_retriever.py  # Simple cosine similarity RAG
│   ├── hybrid_retriever.py    # Dense + Sparse + Reranking
│   ├── evaluator.py          # LLM-as-judge evaluation
│   └── test_queries.py       # 21 evaluation queries
├── research_papers/          # 10 LLM research papers
├── chroma_db/               # Vector database (created after ingestion)
├── outputs/                 # Evaluation results
└── requirements.txt
```

## Quick Run (All Steps)
```bash
python rag_system/data_ingestion.py && python rag_system/evaluator.py
```

## Evaluation Metrics
- **Precision@5**: Percentage of top 5 retrieved docs that are relevant
- **NDCG@5**: Normalized ranking quality of top 5 results
- **LLM Judge**: Uses Groq's Llama-3.1-8b-instant for document relevance scoring

