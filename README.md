# Security Footage Analysis System

A two-part local AI system for analysing security footage using NVIDIA LocateAnything-3B and Gemma 3 4B.

## Part 1 — Perception Pipeline
Upload a video → extract frames → detect & track objects → save to PostgreSQL → view summary report.

## Part 2 — RAG Q&A
Select an analysed video → embed events → ask natural language questions answered by Gemma 3 4B via Ollama.

## Stack
- **Vision model**: [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) via [NVlabs/Eagle](https://github.com/NVlabs/Eagle)
- **LLM**: Gemma 3 4B via Ollama
- **Embeddings**: nomic-embed-text via Ollama (768-dim)
- **Database**: PostgreSQL
- **Framework**: Flask + LangChain LCEL

## Setup

### 1. Clone Eagle dependency
```bash
git clone https://github.com/NVlabs/Eagle eagle
cd eagle/Embodied && pip install -e . && cd ../..
```

### 2. Create virtual environment and install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r surveillance_app/requirements.txt
```

### 3. Pull Ollama models
```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

### 4. Configure environment
```bash
copy surveillance_app\.env.example surveillance_app\.env
# Edit .env with your DATABASE_URL and HF_TOKEN
```

### 5. Create database tables
```bash
cd surveillance_app
python setup_db.py
python part2/migrate_db.py
```

### 6. Start Ollama (separate terminal)
```bash
ollama serve
```

### 7. Run the app
```bash
cd surveillance_app
python app.py
```

Open **http://localhost:5050** in your browser.

- `http://localhost:5050` — Part 1: upload video, view analysis
- `http://localhost:5050/qa` — Part 2: Q&A on analysed footage
