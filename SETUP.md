# RAG System Setup Guide

## Prerequisites
- Python 3.8+
- ~2GB disk space for models and vector database
- GROQ API key (free tier available at https://console.groq.com)

## Installation

### Setup Environment
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
python3 ingestion_pipeline.py
```
This processes the 10 research papers and creates the vector database.

### 5. Test Individual Approaches

**Baseline RAG:**
```bash
python3 baseline_approach/rag_pipeline.py
```

**Improved RAG:**
```bash
python3 improved_approach/rag_pipeline.py
```

### 6. Run Complete Evaluation
```bash
python3 evaluation/rag_evaluator.py
```
This compares both approaches and generates `evaluation_report.md`.

