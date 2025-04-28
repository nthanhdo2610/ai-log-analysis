# AI Log Analysis Project

🚀 An AI-powered FastAPI backend that enables log inspection, vector storage, and AI conversational analysis — running locally with Docker (for Qdrant) and using **Langfuse Cloud** for observability!

This project integrates:
- **Elasticsearch** for searching logs
- **Qdrant** for local vector storage
- **Langfuse Cloud** for AI observability and prompt tracking
- **Sentence Transformers** for text embedding
- **FastAPI** for serving APIs

✅ Only Qdrant runs locally — Langfuse Cloud is used for observability!

---

## 📆 Project Structure

```bash
ai-log-analysis/
 ├── chatbot/         # Chatbot server using FastAPI
 ├── embed/           # Embedding API server (optional)
 ├── utils/           # Config manager and utilities
 ├── dist/            # Build artifacts
 ├── docker-compose.yml  # Docker Compose to start Qdrant
 ├── configs.json     # Configuration settings
 ├── LICENSE
 ├── pyproject.toml   # Python project configuration
 └── README.md        # This documentation
```

---

## ⚙️ Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Python 3.13 or higher
- `pip`, `build`, `wheel` installed

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/nthanhdo2610/ai-log-analysis.git
cd ai-log-analysis
```

---

### 2. Set up Python environment

```bash
python3 -m venv mlenv
source mlenv/bin/activate
pip install --upgrade pip
pip install -e .
```

(Or install dependencies manually from `pyproject.toml`.)

---

### 3. Start Qdrant (local vector database)

```bash
docker compose up -d
```

- Qdrant REST API: [http://localhost:6333](http://localhost:6333)

✅ Langfuse is used via [https://cloud.langfuse.com](https://cloud.langfuse.com)

---

### 4. Run Embedding API Server (Optional)

```bash
python embed/embedded_api.py
```

Available at:

```
POST http://localhost:8080/embed
```

Takes raw text and returns vector embeddings.

---

### 5. Run Chatbot Server

```bash
python chatbot/chat_api.py
```

FastAPI server available at:

- Root: [http://localhost:8000/](http://localhost:8000/)
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🛠️ Configuration

The system loads configuration in two ways:
1. From **environment variables** (highest priority)
2. Fallback to **`configs.json`** if environment variables are missing.

Example `configs.json`:

```json
{
  "es_config": {
    "es_host": "https://127.0.0.1:9200",
    "es_user": "elastic",
    "es_pass": "changeme",
    "index": "logs-local"
  },
  "qdrant_db": {
    "host": "127.0.0.1",
    "port": 6333,
    "collection_name": "logs_embed",
    "prefer_grpc": true,
    "https": false
  },
  "embedding_model": "http://localhost:8080/embed/",
  "langfuse_keys": {
    "secret_key": "your-langfuse-secret-key",
    "public_key": "your-langfuse-public-key",
    "host": "https://cloud.langfuse.com"
  },
  "claude_config": {
    "claude_model": "claude-3-opus",
    "claude_api_key": "sk-ant-test"
  }
}
```

✅ Easy switching between development, staging, and production!

---

## 📚 Useful Commands

| Task | Command |
|:-----|:--------|
| Start Qdrant service | `docker compose up -d` |
| Stop all running services | `docker compose down` |
| Build Python project | `python -m build` |
| Install project locally | `pip install -e .` |
| Run Embedding server | `python embed/embedded_api.py` |
| Run Chatbot FastAPI server | `python chatbot/chat_api.py` |

---

## 🧐 Features

- 🚀 FastAPI-based API server for chatbot interaction
- 🔎 Elasticsearch for structured log search
- 🧐 Sentence Transformers for text/vector embeddings
- 🔥 Langfuse Cloud for prompt observability
- 🗂️ Qdrant Vector DB for fast document retrieval
- 🖥️ Minimal local services using Docker Compose
- ⚡ Easy plug-and-play configuration (configs.json + ENV)

---

## 🛡️ Security Note

- This project is for **development and testing**.
- For production, you should enable:
  - SSL/HTTPS for APIs
  - Authentication (OAuth2, API keys)
  - Secure Elasticsearch/Qdrant
  - Protect your Langfuse API keys

---

## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for full details.

---

## ✨ Future Enhancements

- Add WebSocket real-time log ingestion
- Integrate OpenAI or Anthropic models for fallback
- Support JWT authentication
- Improve API validation and error handling
- Build deployment templates (AWS ECS / Kubernetes)

---

# 🚀 Happy Hacking & AI Log Adventures!

