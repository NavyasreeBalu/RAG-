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
echo "MODEL_NAME=llama3-8b-8192" >> .env
```

## Usage

### 4. Build Knowledge Base 
```bash
python src/data_ingestion.py
```
This processes the 10 research papers and creates the vector database.

### 5. Test Individual Approaches

**Baseline RAG:**
```bash
python src/baseline_retriever.py
```

**Hybrid RAG:**
```bash
python src/hybrid_retriever.py
```

### 6. Run Complete Evaluation
```bash
python src/evaluator.py
```
This compares both approaches and generates `outputs/evaluation_report.md`.

### 7. View Results
```bash
cat outputs/evaluation_report.md
```

## Quick Run (All Steps)
```bash
python src/data_ingestion.py && python src/evaluator.py
```

