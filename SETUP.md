# RAG System Setup

## Installation

1. **Clone repository**:
   ```bash
   git clone <your-repo>
   cd gen101
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements-clean.txt
   ```

4. **Setup environment**:
   ```bash
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

5. **Run baseline approach**:
   ```bash
   cd baseline_approach
   python3 ingestion_pipeline.py  # Run once
   python3 rag_pipeline.py        # Run RAG system
   ```

## Dependencies
- Python 3.8+
- Google AI API key (free tier available)
- ~2GB disk space for models and vector database
