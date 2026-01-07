# RAG System Setup Guide

## Prerequisites
- Python 3.0+
- GROQ API key (free tier available at https://console.groq.com)

## Installation

### 1. Setup Environment
```bash
cd RAG
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
├── README.md                 # Project overview and approach
├── SETUP.md                 # Installation and usage instructions
├── results.md               # Evaluation results and analysis
├── problem_statement.md     # Original problem definition
├── requirements.txt         # Python dependencies
├── rag_system/             # Core implementation
│   ├── baseline_retriever.py    # Simple cosine similarity baseline
│   ├── hybrid_retriever.py      # Hybrid retrieval implementation
│   ├── data_ingestion.py        # Document processing and indexing
│   ├── evaluator.py             # Evaluation framework
│   └── test_queries.py          # Test query definitions
├── research_papers/        # LLM research paper corpus (10 papers)
├── chroma_db/             # Vector database storage
└── outputs/               # Detailed evaluation reports
```

## Project Structure

## Quick Run (All Steps)
```bash
python rag_system/data_ingestion.py && python rag_system/evaluator.py
```
