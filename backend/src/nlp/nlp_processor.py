"""
NLP Processing Engine for Limier AML Risk Scoring — by Prakhar Pandey.

Two-tier approach:
  1. HuggingFace Inference API (when HF_TOKEN is set):
       - Zero-shot typology classification via facebook/bart-large-mnli
       - Sentiment analysis via distilbert-base-uncased-finetuned-sst-2-english
  2. Fallback: keyword TF-IDF weighted scoring (runs fully offline, no API needed).

The HF calls are wrapped in try/except so the server never crashes even if the
API is unreachable or rate-limited.
"""

from __future__ import annotations
import re
import os
import math
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AML Keyword dictionary (used for fallback scoring AND keyword extraction)
# ---------------------------------------------------------------------------

AML_KEYWORDS: Dict[str, float] = {
    "offshore": 0.85,
    "shell": 0.90,
    "structuring": 0.95,
    "smurf": 0.90,
    "crypto": 0.70,
    "bitcoin": 0.70,
    "usdt": 0.75,
    "consulting": 0.50,
    "loan repayment": 0.60,
    "gift": 0.55,
    "urgent": 0.65,
    "cash": 0.75,
    "invoice": 0.40,
    "unregistered": 0.85,
    "hawala": 0.95,
    "sanction": 0.95,
    "nominee": 0.80,
    "wire transfer": 0.35,
    "overseas": 0.60,
    "anonymous": 0.90,
}

# AML typology labels for zero-shot classification
AML_TYPOLOGY_LABELS = [
    "Structuring & Cash Smurfing",
    "Offshore Shell & Layering",
    "High-Risk Digital Asset Velocity",
    "Rapid Cash-Out & Layering",
    "Trade-Based Money Laundering",
    "Hawala & Informal Value Transfer",
    "Standard Retail Transaction",
]

# ---------------------------------------------------------------------------
# Synthetic memo pools
# ---------------------------------------------------------------------------

HIGH_RISK_MEMOS = [
    "Urgent wire transfer to offshore shell consulting entity",
    "Loan repayment cash deposit below threshold",
    "Crypto OTC transfer to anonymous wallet",
    "Invoice payment for unregistered overseas advisory services",
    "Split structuring deposit for nominee account",
    "Rapid cash withdrawal - urgent family gift",
    "Hawala arrangement for overseas family remittance",
    "Bitcoin purchase via cash OTC desk",
    "Anonymous bearer bond redemption through nominee",
]

NORMAL_MEMOS = [
    "Monthly payroll credit",
    "Grocery store debit purchase",
    "Utility bill payment",
    "Internal checking account transfer",
    "Standard vendor payment for office supplies",
    "Online retail order payment",
    "ATM withdrawal for personal use",
    "Direct deposit from employer",
]

# ---------------------------------------------------------------------------
# HuggingFace Inference API helpers
# ---------------------------------------------------------------------------

_HF_BASE = "https://api-inference.huggingface.co/models"
_hf_session = None  # lazily initialized httpx session


def _get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")


def _hf_headers() -> dict:
    token = _get_hf_token()
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _hf_zero_shot_classification(text: str, labels: List[str]) -> Dict[str, Any] | None:
    """
    Calls HuggingFace zero-shot classification endpoint.
    Returns {'labels': [...], 'scores': [...]} or None on failure.
    """
    if not _get_hf_token():
        return None
    try:
        import httpx
        payload = {
            "inputs": text,
            "parameters": {"candidate_labels": labels, "multi_label": False},
        }
        resp = httpx.post(
            f"{_HF_BASE}/facebook/bart-large-mnli",
            headers=_hf_headers(),
            content=json.dumps(payload),
            timeout=10.0,
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 503:
            # Model loading — try a quick fallback model
            resp2 = httpx.post(
                f"{_HF_BASE}/cross-encoder/nli-deberta-v3-small",
                headers=_hf_headers(),
                content=json.dumps(payload),
                timeout=10.0,
            )
            if resp2.status_code == 200:
                return resp2.json()
        logger.warning(f"HF zero-shot API returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"HF zero-shot call failed: {e}")
    return None


def _hf_sentiment(text: str) -> Dict[str, Any] | None:
    """
    Calls HuggingFace sentiment analysis endpoint.
    Returns {'label': 'POSITIVE'|'NEGATIVE', 'score': float} or None on failure.
    """
    if not _get_hf_token():
        return None
    try:
        import httpx
        resp = httpx.post(
            f"{_HF_BASE}/distilbert-base-uncased-finetuned-sst-2-english",
            headers=_hf_headers(),
            content=json.dumps({"inputs": text[:512]}),
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Returns [[{label, score}, ...]] — take highest
            if isinstance(data, list) and data and isinstance(data[0], list):
                best = max(data[0], key=lambda x: x["score"])
                return best
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                return max(data, key=lambda x: x["score"])
        logger.warning(f"HF sentiment API returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"HF sentiment call failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Core NLP analysis
# ---------------------------------------------------------------------------

def analyze_text_narrative(memos: List[str]) -> Dict[str, Any]:
    """
    Analyzes a list of transaction text memos/narratives for a customer.
    Uses HuggingFace Inference API when available, otherwise falls back to
    keyword-weighted scoring.

    Returns NLP score, sentiment, urgency, detected keywords, typology, sample_memos.
    """
    if not memos:
        return {
            "nlp_suspicious_score": 0.0,
            "sentiment": "neutral",
            "urgency_level": "low",
            "detected_keywords": [],
            "typology": "Standard Activity",
            "sample_memos": [],
        }

    combined_text = " ".join(memos).lower()
    words = re.findall(r"\b[a-z0-9_]+\b", combined_text)

    # 1. Keyword extraction (always run — feeds into scoring and tag cloud)
    detected_keywords: List[str] = []
    total_weight = 0.0
    for kw, weight in AML_KEYWORDS.items():
        if kw in combined_text:
            detected_keywords.append(kw.title())
            total_weight += weight

    # 2. Urgency heuristic (deterministic, cheap)
    urgent_words = {"urgent", "immediate", "asap", "fast", "overnight", "express"}
    negative_words = {"unregistered", "anonymous", "shell", "sanction", "smurf", "hawala"}
    urgency_count = sum(1 for w in words if w in urgent_words)
    negative_count = sum(1 for w in words if w in negative_words)
    urgency_level = "high" if urgency_count > 0 else "low"

    # 3. Try HuggingFace zero-shot typology classification
    typology = None
    hf_score_boost = 0.0
    zs_result = _hf_zero_shot_classification(combined_text[:512], AML_TYPOLOGY_LABELS)
    if zs_result and "labels" in zs_result and "scores" in zs_result:
        top_label = zs_result["labels"][0]
        top_score = zs_result["scores"][0]
        # Only use if not "standard" and confident enough
        if top_label != "Standard Retail Transaction" and top_score > 0.35:
            typology = top_label
            hf_score_boost = top_score * 0.3  # HF confidence boosts suspicion score
        elif top_label == "Standard Retail Transaction" and top_score > 0.60:
            typology = "Standard Retail Transaction"
        else:
            typology = None  # fall through to keyword-based

    # 4. Try HuggingFace sentiment analysis
    sentiment = None
    hf_sentiment_result = _hf_sentiment(combined_text[:512])
    if hf_sentiment_result:
        raw_label = hf_sentiment_result.get("label", "").upper()
        conf = hf_sentiment_result.get("score", 0.5)
        if raw_label == "NEGATIVE" and conf > 0.6:
            sentiment = "negative"
        elif raw_label == "POSITIVE" and conf > 0.7:
            sentiment = "neutral"  # positive transactions are normal
        else:
            sentiment = "cautious"

    # 5. Fallback typology (keyword-based)
    if typology is None:
        if "Structuring" in detected_keywords or "Smurf" in detected_keywords or "Cash" in detected_keywords:
            typology = "Structuring & Cash Smurfing"
        elif "Offshore" in detected_keywords or "Shell" in detected_keywords or "Overseas" in detected_keywords:
            typology = "Offshore Shell & Layering"
        elif "Crypto" in detected_keywords or "Bitcoin" in detected_keywords or "Usdt" in detected_keywords:
            typology = "High-Risk Digital Asset Velocity"
        elif "Hawala" in detected_keywords:
            typology = "Hawala & Informal Value Transfer"
        elif detected_keywords:
            typology = f"Unusual Narrative Pattern ({', '.join(detected_keywords[:2])})"
        else:
            typology = "Standard Retail Transaction Text"

    # 6. Fallback sentiment (keyword-based)
    if sentiment is None:
        sentiment = (
            "negative" if negative_count > 0
            else ("cautious" if urgency_count > 0 else "neutral")
        )

    # 7. Calculate normalized suspicious score (0.0 – 1.0)
    base_score = min(1.0, total_weight / 2.0)
    # Blend keyword score with HF confidence boost
    nlp_suspicious_score = round(min(1.0, base_score + hf_score_boost), 3)

    return {
        "nlp_suspicious_score": float(nlp_suspicious_score),
        "sentiment": sentiment,
        "urgency_level": urgency_level,
        "detected_keywords": detected_keywords,
        "typology": typology,
        "sample_memos": memos[:3],
    }


# ---------------------------------------------------------------------------
# Memo generator
# ---------------------------------------------------------------------------

def generate_customer_memos(customer_id: str, is_suspicious: bool = False, count: int = 5) -> List[str]:
    """Generates synthetic transaction memos for customer text analysis."""
    import random
    seed_val = int(abs(hash(customer_id))) % 10000
    rng = random.Random(seed_val)

    memos = []
    for _ in range(count):
        if is_suspicious or rng.random() < 0.35:
            memos.append(rng.choice(HIGH_RISK_MEMOS))
        else:
            memos.append(rng.choice(NORMAL_MEMOS))
    return memos
