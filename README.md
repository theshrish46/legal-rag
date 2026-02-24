# ⚖️ Legal AI Auditor

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)
![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-red)
![Gemini](https://img.shields.io/badge/AI-Gemini_1.5_Flash-8E75B2)

A professional-grade **Retrieval-Augmented Generation (RAG)** application designed for auditing complex legal documents. Built with a robust containerized architecture, it moves beyond simple chat to provide **Deep Contextual Audits** of contracts using "Lost-in-the-Middle" mitigation strategies.

## Overview

This auditor uses a state-of-the-art RAG pipeline to allow legal professionals to interact with large volumes of documents. It features recursive text splitting optimized for legal clauses and a streaming UI for a real-time, low-latency experience.

*   **Brain:** Gemini 1.5 Flash (via LangChain Google GenAI)
*   **Memory:** Qdrant Vector Database (Local Docker Instance)
*   **Interface:** Streamlit (with Session State Memory)
*   **Infrastructure:** Docker & Docker Compose

## Key Engineering Features

*   **Deep Contextual Audit:** Implemented `RecursiveCharacterTextSplitter` with **2000-character chunks** and high overlap to ensure legal definitions and multi-paragraph clauses remain semantically intact.
*   **Metadata Injection:** Custom ingestion pipeline extracts document metadata (Filename, Date, Type) and injects it directly into vector embeddings to improve retrieval accuracy.
*   **Persistent Vector Memory:** Integrated with **Qdrant** to store and retrieve document embeddings efficiently, preventing re-indexing of duplicate files.
*   **Real-time Streaming:** Implemented token-by-token response streaming using LangChain generators for a responsive user experience.
*   **Dockerized Architecture:** Fully containerized microservices architecture ensuring consistent performance across development and production environments.

## Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Orchestration** | LangChain (LCEL) | Manages the Retrieval/Generation chains. |
| **Frontend** | Streamlit | Chat interface with `st.session_state` history. |
| **Vector Store** | Qdrant | High-performance vector similarity search. |
| **LLM** | Google Gemini 1.5 Flash | Selected for 1M+ token context window. |
| **DevOps** | Docker Compose | Multi-container orchestration. |

## Installation & Setup

### 1. Prerequisites
*   Docker and Docker Compose installed.
*   A Google Gemini API Key.

### 2. Environment Configuration
Create a `.env` file in the root directory:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

### 3. Deploy with Docker
Launch the entire stack from scratch to ensure a clean build:
```bash
# Build and start the containers
docker compose up --build -d
```

## Usage

1.  **Access:** Open the UI at `http://localhost:8501`.
2.  **Upload:** Use the sidebar to upload PDF legal contracts.
3.  **Audit:** Ask complex reasoning questions:
    *   *"What are the termination liabilities in Section 4.2?"*
    *   *"Is there a non-compete clause that exceeds 12 months?"*
4.  **Monitor:** To check the logs in real-time:
    ```bash
    docker compose logs -f
    ```

## Project Structure

```text
legal-rag/
├── src/
│   ├── database/       # Qdrant connection & duplicate check logic
│   ├── retriever/      # Legal-specific retrieval strategies
│   ├── prompts/        # Strict Legal System Prompts
│   ├── utils/          # Chat history and formatting utilities
├── main.py             # Streamlit Entry Point
├── Dockerfile          # App container definition
└── docker-compose.yml  # Multi-container orchestration
```

## 🧹 Maintenance

To stop the application and clear the cache/volumes (useful for resetting the database):

```bash
docker compose down
docker system prune -a --volumes
```

---
