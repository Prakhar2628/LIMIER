# Limier — AI-Powered AML Investigation Platform

<p align="center">
 <img width="1774" height="887" alt="Limier Banner" src="https://github.com/user-attachments/assets/e897022c-5209-4547-9a02-f1c938260f3f" />
</p>

<p align="center">
  <b>An autonomous, explainable AI agent for anti-money laundering detection and investigation.</b><br/>
  <sub>Built by <strong>Prakhar Pandey</strong> · <a href="https://github.com/Prakhar2628/LIMIER">github.com/Prakhar2628/LIMIER</a></sub>
</p>

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Screenshot](#screenshot)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Solution Approach](#solution-approach)
- [Model Training & Caching](#model-training--caching)
- [Dataset Information](#dataset-information)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Data Sources](#data-sources)
- [Notes & Limitations](#notes--limitations)
- [Author](#author)

---

## Problem Statement

> **AI-Powered Suspicious Activity Detection**
>
> Financial institutions are required to run robust Anti-Money Laundering
> (AML) compliance programs, but traditional rule-based transaction
> monitoring systems generate a high volume of false positives, overwhelming
> compliance teams while sophisticated laundering techniques — structuring,
> smurfing, layering — continue to slip through.

---

## Screenshot

<p align="center">
 <img width="1917" height="872" alt="Limier Dashboard" src="https://github.com/user-attachments/assets/1596d450-f694-4081-abbf-09581d8856c5" />
</p>

---

## Architecture

<p align="center">
  <img width="1024" height="1536" alt="Architecture Diagram" src="https://github.com/user-attachments/assets/0ab46e7a-b4ce-4a0c-940f-32ef7f2ae23b" />
</p>

At a high level, Limier is a **3-pillar hybrid detection engine** wrapped in
an autonomous LLM agent, sitting behind a FastAPI backend and a Next.js
dashboard:

---

## Tech Stack

**Frontend**
- Next.js 16 (App Router), React 19, TypeScript
- Tailwind CSS v4 + Glassmorphism design system
- Framer Motion (animation), Recharts (data visualization)
- Radix UI primitives, lucide-react (icons), react-markdown

**Backend**
- FastAPI + Uvicorn (ASGI)
- Python 3.11
- Pandas, NumPy (data processing)
- Scikit-learn (Isolation Forest), XGBoost 1.7.6 (supervised classification)
- SHAP (SHapley Additive exPlanations) for explainable AI
- Joblib (model serialization & disk caching for sub-second startup)
- Pytest (testing)

**AI Agent / LLM Layer**
- Native function/tool calling (`qwen/qwen3.6-27b` on Groq API) — no LangChain overhead
- 6 Native Tools (`get_top_high_risk_customers`, `get_customer_profile`, `get_dataset_statistics`, `generate_sar`, `get_customer_transactions`, `analyze_transactions_dataframe`)
- **NLP Engine**: Hugging Face Inference API (`distilbert-base-uncased-finetuned-sst-2-english`) for zero-shot AML typology classification and sentiment analysis
- **Generative AI**: LLM-powered risk narratives panel & formal markdown SAR generator with transaction evidence

**Infrastructure**
- Docker & Docker Compose (single-command startup for both services)
- python-dotenv for environment configuration

---

## Solution Approach

Limier avoids relying on a single black-box model. It combines three
independent detection methodologies — each contributing a different kind of
signal — into one final, fully explainable risk score:

1. **Deterministic Rules Engine** — catches known AML typologies (structuring,
   rapid cash-out, round-tripping, velocity spikes, dormant-then-active
   accounts, high-risk geography, round-number bias) with zero uncertainty.
   Thresholds are config-driven, not hardcoded, and every rule produces a
   human-readable reason string. A high-severity rule firing **overrides**
   the ML score and immediately elevates a customer to High Risk.

2. **Unsupervised ML (Isolation Forest)** — detects novel, previously unseen
   anomalies in the transaction feature space without needing labeled data,
   catching laundering patterns that don't match a known rule.

3. **Supervised ML (XGBoost) + SHAP** — learns from labeled synthetic fraud
   patterns and produces per-transaction SHAP feature attributions, so every
   ML-driven flag can be traced back to the specific features that drove it.

4. **Stacked MetaEnsemble Calibration** — calibrates probabilities using a 
   Platt logistic meta-learner trained on out-of-sample 80/20 validation split.

On top of this scoring engine sits an **LLM investigation agent** powered by 
`qwen/qwen3.6-27b` that dynamically parses natural language queries (e.g. *"Find structuring patterns in the last 30 days"* or *"Show top 5 suspicious customers and generate a SAR"*) and selectively invokes tools to execute requests with streaming trace logs.

---

## Model Training & Caching

To prevent re-parsing 200,000+ transactions and re-building features on every server restart, Limier includes:
- **Jupyter Notebook**: [`backend/notebooks/train_model.ipynb`](file:///d:/Limier/backend/notebooks/train_model.ipynb) — interactive notebook to train models, evaluate AUC-PR metrics, and pre-compute feature matrices.
- **CLI Training Script**: [`backend/scripts/train_and_cache.py`](file:///d:/Limier/backend/scripts/train_and_cache.py) — script that trains models and outputs `risk_summary_df.pkl` to `backend/data/cache/`.
- **Sub-second Fast Startup**: FastAPI automatically detects pre-computed cache files, dropping server startup time from **69.4 seconds to 0.67 seconds**.

---

## Dataset Information

Since real banking transaction data is private and cannot be used, Limier
generates a **realistic synthetic dataset at startup**:

- **Scale:** ~2,000 customers, ~211,000 transactions
- **Realism mechanics:**
  - Log-normal transaction amount distributions (heavy-tailed, matching real
    financial data)
  - Poisson-process transaction timing, biased toward business hours
  - 80% counterparty "stickiness" (simulating recurring rent/salary/grocery
    payments rather than a random network)
  - Mostly domestic (US) activity with a realistic long tail of cross-border
    transfers
- **Typology injection:** a small subset of customers have explicitly
  planted AML patterns (structuring, rapid cash-out, round-tripping,
  dormant-then-active) with ground-truth labels (`is_planted_suspicious`,
  `planted_pattern`) used strictly for model evaluation.

---

## Quick Start

**Prerequisites:** Docker Desktop installed and running.

### 1. Set up environment variables

Copy `env-vars.txt` to `.env`:

```bash
cp env-vars.txt .env
```

*(On Windows PowerShell: `Copy-Item env-vars.txt .env`)*

Ensure your `.env` contains:
```env
LLM_PROVIDER=groq
LLM_MODEL=qwen/qwen3.6-27b
GROQ_API_KEY=gsk_...
HF_TOKEN=hf_...
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 2. Build and run

From the project root:

```bash
docker compose up --build
```

### 3. Open the app

- **Frontend dashboard:** [http://localhost:3000](http://localhost:3000)
- **Backend API:** [http://localhost:8000](http://localhost:8000)
- **Health check:** [http://localhost:8000/health](http://localhost:8000/health)

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | No | `groq`, `huggingface`, `xai`, or `openai` — selects LLM backend |
| `LLM_API_KEY` / `GROQ_API_KEY` | No | API key for Groq LLM service |
| `HF_TOKEN` | No | Hugging Face token for Inference API NLP sentiment & typology model |
| `LLM_MODEL` | No | Model name (`qwen/qwen3.6-27b`) |
| `NEXT_PUBLIC_API_URL` | Yes | Backend URL (`http://localhost:8000`) |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Returns API health and cache status |
| `GET` | `/eda` | Exploratory data analysis: distributions, breakdowns, missing values |
| `GET` | `/customers/{customer_id}/risk` | Pre-computed risk profile for a single customer |
| `GET` | `/customers/{customer_id}/xai` | Fresh SHAP feature attributions for XAI drawer |
| `GET` | `/customers/{customer_id}/narrative` | LLM-generated plain English risk narrative |
| `POST` | `/score` | Score raw transactions, or filter cached data by customer/date/pattern |
| `POST` | `/api/agent/query` | Streaming (SSE) endpoint for the LLM investigation agent |

Full interactive API docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Project Structure

```
Limier/
├── backend/
│   ├── notebooks/            # Jupyter training & pre-computation notebook
│   ├── scripts/              # CLI model training & caching script
│   ├── src/
│   │   ├── api/              # FastAPI app, routes, request/response models
│   │   ├── agent/            # LLM query agent, tool definitions, SAR generator
│   │   ├── nlp/              # Hugging Face NLP inference processor & typology extractor
│   │   ├── models/           # Rules engine, Isolation Forest, XGBoost, MetaEnsemble, SHAP
│   │   ├── features/         # Feature engineering (point-in-time, rolling windows)
│   │   └── data/             # Synthetic data generator
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/               # Next.js App Router pages
│   │   ├── components/        # Dashboard, chat panel, trace log, SHAP charts, drawer
│   │   └── lib/               # API client with retry, types, SSE stream helper
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml
├── env-vars.txt              # Copy to .env before first run
└── README.md
```

---

## Usage Examples

Once running, try asking the agent (via the dashboard chat panel):

- *"Show me the top 5 most suspicious customers and generate a SAR for the highest risk one."* → triggers multi-step tool call sequence
- *"Give me an overview of this dataset"* → triggers broad EDA
- *"Find structuring patterns in the last 30 days"* → targeted pattern detection
- *"Is customer CUST_00707 suspicious?"* → single-entity lookup and SHAP breakdown

---

## Author

**Prakhar Pandey** — [github.com/Prakhar2628/LIMIER](https://github.com/Prakhar2628/LIMIER)