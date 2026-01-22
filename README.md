# Customer Service Chatbot

A lightweight, CPU‑friendly RAG chatbot built with FAISS, SentenceTransformers, FLAN‑T5, and a Gradio chat UI. The bot is designed to answer customer‑support questions using a small knowledge base while running smoothly on machines without a GPU.

**Project Overview**

This project implements a customer-service assistant capable of:

- Retrieving relevant information from a small FAQ knowledge base

- Converting documents and queries into vector embeddings using all-MiniLM-L6-v2

- Using semantic search via FAISS

- Generating grounded answers using FLAN‑T5‑Base

- Maintaining short‑term conversation memory

- Falling back when no relevant information is found

- Providing a Gradio chat interface

**Quick Start**

- **Install dependencies:**

```bash
pip install -r requirements.txt
```

- **Run the Chatbot:**

```bash
python rag_bot.py
```

**Example Queries**

- "My headphones won't pair, what should I do"

- "My phone won't turn on"

- "My laptop is overheating"

**Next Steps**

- Expand the knowledge base

- Add long‑term memory or summarization

- Improve retrieval accuracy and response relevance

- Enhance the UI