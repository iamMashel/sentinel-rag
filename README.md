# Sentinel RAG: Intelligent Customer Support Engine

A production-ready Retrieval-Augmented Generation (RAG) system engineered for automatically customer support. This project demonstrates a professional approach to building AI agents with **Python**, **Google Gemini**, **ChromaDB**, and **Streamlit**.

## 🚀 Features

- **Hybrid Response Engine**: Intelligently switches between local Knowledge Base retrieval (high precision) and LLM generation (high flexibility) based on confidence thresholds.
- **Scalable Vector Search**: Utilizes **ChromaDB** for efficient, persistent vector storage, replacing basic in-memory numpy operations for production scalability.
- **Interactive UI**: Built with **Streamlit** for real-time interaction and debugging.
- **Professional Engineering**:
    - **Modular Architecture**: Structured as a python package (`sentinel_rag`).
    - **Dependency Management**: Powered by `uv` for lightning-fast and reliable builds.
    - **Type Hinting**: Fully typed codebase for maintainability.
    - **Structured Logging**: Comprehensive logging for system health monitoring.
    - **Environment Security**: API keys managed securely via `.env`.

## 🛠 Tech Stack

- **Language**: Python 3.12+
- **LLM & Embeddings**: Google Gemini (gemini-pro, text-embedding-004)
- **Vector Database**: ChromaDB
- **Frontend**: Streamlit
- **Dependency Manager**: uv
- **Testing**: Pytest

## ⚡ Quick Start

### Prerequisites
- Python 3.8+ installed.
- Google Gemini API Key.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/iamMashel/sentinel-rag.git
    cd sentinel-rag
    ```

2.  **Install dependencies with `uv`:**
    ```bash
    # Install uv if you haven't needed
    pip install uv

    # Sync dependencies
    uv sync
    ```
    *Alternatively, use pip: `pip install -r requirements.txt` (if generated)*

3.  **Configure Environment:**
    ```bash
    cp .env.example .env
    # Edit .env and add your GEMINI_API_KEY
    ```

4.  **Run the Application:**
    ```bash
    uv run streamlit run app.py
    ```

## 🏗 Architecture

The system follows a modular design:

- **src/sentinel_rag/core/engine.py**: Contains the `SupportBot` class, orchestrating the RAG flow.
- **src/sentinel_rag/vector_db/store.py**: Wrapper for ChromaDB interactions.
- **app.py**: Streamlit frontend for user interaction.

## 🧪 Testing

Run production-grade tests to verify system integrity:

```bash
uv run pytest
```

## 📈 Scalability Note

This system uses ChromaDB, a purpose-built vector database. Unlike simple in-memory arrays, this allows the system to scale to millions of documents without performance degradation, making it suitable for enterprise deployments.

---
**Built by Mashel**
