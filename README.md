---
title: NovaDXB
emoji: 📈
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: AI-powered Agentic Concierge for Dubai Tourists and Visitors
---

<div align="center">

# ✦ NOVADXB

### Explore Dubai. Intelligently.

**An AI agentic concierge that plans your entire Dubai trip — itinerary, budget, dining, and area recommendations — through natural conversation.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/nipunkavindaAI/NovaDXB)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.3-1C3C3C?style=for-the-badge)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[**🚀 Try the Live App**](https://huggingface.co/spaces/nipunkavindaAI/NovaDXB) · [**📋 Report a Bug**](https://github.com/NipunKavinda95/NovaDXB/issues) · [**💡 Request a Feature**](https://github.com/NipunKavinda95/NovaDXB/issues)

</div>

---

## 📖 Overview

Dubai welcomes **18+ million tourists every year**, yet trip planning still means scattered blog posts, outdated listicles, and guesswork about real costs.

**NovaDXB** is an agentic AI concierge built specifically for Dubai tourism. Rather than a simple Q&A chatbot, it's a reasoning agent equipped with specialized tools that plans complete, personalized Dubai experiences — thinking like a local concierge, not a search engine.

> Tell it your budget, your travel style, and your dates — NovaDXB builds a complete itinerary with real area names, real restaurant recommendations, and real AED pricing, pulled from a curated Dubai-specific knowledge base.

---

## ✨ Key Features

| Feature                        | Description                                                                       |
| ------------------------------ | --------------------------------------------------------------------------------- |
| 🗺️ **Smart Itinerary Builder** | Generates complete day-by-day Dubai plans with real places, timing, and costs     |
| 🏨 **Area Recommender**        | Matches the best Dubai neighborhood to your budget and travel style               |
| 🍽️ **Dining Concierge**        | Restaurant recommendations from street food to fine dining, with real AED pricing |
| 💰 **Budget Estimator**        | Transparent daily/total cost breakdowns across Budget, Mid, and Luxury tiers      |
| 🌤️ **Weather Advisor**         | Seasonal guidance and packing tips based on Dubai's climate patterns              |
| 💱 **Currency Converter**      | Quick AED conversions for major tourist currencies                                |
| 📍 **Local Knowledge Base**    | Curated insider tips and hidden gems most tourists never discover                 |
| 🤖 **Agentic Reasoning**       | Multi-step planning — the agent decides which tools to call, not a fixed script   |

---

## 🛠️ Tech Stack

<div align="center">

| Layer                | Technology                                  |
| -------------------- | ------------------------------------------- |
| **LLM**              | OpenAI GPT-4o-mini                          |
| **Agent Framework**  | LangChain + LangGraph (ReAct agent pattern) |
| **RAG Engine**       | LlamaIndex                                  |
| **Vector Database**  | Pinecone                                    |
| **Backend**          | Flask + Waitress (WSGI)                     |
| **Frontend**         | HTML / CSS / JavaScript                     |
| **Containerization** | Docker                                      |
| **Deployment**       | HuggingFace Spaces                          |

</div>

### Architecture

```
┌─────────────┐      ┌──────────────┐      ┌──────────────────┐
│   Frontend   │ ───▶ │  Flask API    │ ───▶ │  LangGraph Agent  │
│ (HTML/CSS/JS)│      │ (rate-limited,│      │   (7 MCP tools)   │
└─────────────┘      │  sanitized)   │      └─────────┬─────────┘
                       └──────────────┘                │
                                                         ▼
                                               ┌──────────────────┐
                                               │   RAG Engine      │
                                               │  (LlamaIndex)     │
                                               └─────────┬────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────┐
                                               │  Pinecone Vector   │
                                               │     Database       │
                                               └──────────────────┘
```

The agent receives a user query, reasons about which tool(s) it needs (itinerary planning, budget estimation, area recommendation, etc.), retrieves grounded context from the Dubai knowledge base via RAG, and returns a specific, actionable response — never generic advice.

---

## 📂 Project Structure

```
NovaDXB/
├── data/                      # Knowledge base (RAG source documents)
│   ├── areas_guide.csv        # 20+ Dubai neighborhoods
│   ├── attractions.csv        # 30+ attractions with costs & tips
│   ├── dining_guide.csv       # 25+ restaurants across all budgets
│   ├── practical_info.txt     # Visa, transport, culture, safety
│   ├── budget_logic.txt       # Budget/Mid/Luxury cost breakdowns
│   ├── seasonal_guide.txt     # Month-by-month climate & events
│   └── hidden_gems.txt        # Local insider tips
├── static/                    # Frontend assets
│   ├── index.html             # Chat interface
│   └── landing.html           # Landing page
├── app.py                     # Flask application (routes, security, rate limiting)
├── agent.py                   # LangGraph ReAct agent + 7 MCP tools
├── rag_engine.py               # LlamaIndex + Pinecone RAG pipeline
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container definition
└── .env.example                 # Environment variable template
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/)
- A [Pinecone API key](https://www.pinecone.io/) with an index created (1536 dimensions, cosine metric)

### Installation

```bash
# Clone the repository
git clone https://github.com/NipunKavinda95/NovaDXB.git
cd NovaDXB

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=novadxb
SECRET_KEY=generate_with_python_secrets_token_hex_32
INGEST=false
```

> Generate a secure `SECRET_KEY` with: `python -c "import secrets; print(secrets.token_hex(32))"`

### First-Time Setup — Ingest the Knowledge Base

```bash
# Set INGEST=true in .env, then run once:
python rag_engine.py

# Set INGEST=false afterward to avoid re-ingesting on every restart
```

### Run Locally

```bash
python app.py
```

Visit `http://localhost:7860` to see the landing page, or `http://localhost:7860/app` for the chat interface.

---

## 🔒 Security & Reliability

NovaDXB was built with production-grade practices, not just hackathon-grade ones:

- ✅ **Input sanitization** — strips control characters, enforces length limits
- ✅ **Prompt-injection detection** — flags and safely redirects override attempts
- ✅ **Rate limiting** — per-IP sliding window, prevents abuse and runaway API costs
- ✅ **Response caching** — identical queries served instantly, reduces LLM calls
- ✅ **Restricted CORS** — scoped to known origins, not wide open
- ✅ **No leaked exceptions** — full error details logged server-side only
- ✅ **Graceful startup handling** — missing API keys fail safely, not silently

---

## 🐳 Docker Deployment

```bash
docker build -t novadxb .
docker run -p 7860:7860 --env-file .env novadxb
```

---

## 🗺️ Roadmap

- [x] Core RAG pipeline with Pinecone + LlamaIndex
- [x] Agentic reasoning with 7 specialized tools
- [x] Security hardening (rate limiting, sanitization, CORS)
- [x] Live deployment on HuggingFace Spaces
- [ ] Premium split-panel UI with live itinerary visualization
- [ ] Interactive map integration
- [ ] Multi-language support

---

## 🤝 Acknowledgments

Built as part of the **Decoding Data Science — AI Application Building Challenge**, supported by JetBrains and Google for Developers.

---

## 📬 Connect

**Nipun Kavinda** — Industrial AI & MLOps Engineer, Dubai

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/nipun-kavinda/)
[![GitHub](https://img.shields.io/badge/GitHub-NipunKavinda95-181717?style=flat&logo=github)](https://github.com/NipunKavinda95)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-nipunkavindaAI-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/nipunkavindaAI)

---

<div align="center">

_If you found this project interesting, consider giving it a ⭐_

</div>
