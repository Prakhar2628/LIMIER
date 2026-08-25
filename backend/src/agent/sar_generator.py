"""
AI Agent for generating Suspicious Activity Reports (SAR) and Risk Narratives.
Supports Groq, HuggingFace Inference API, xAI (Grok), and OpenAI-compatible endpoints.
Author: Prakhar Pandey
"""
import os
import sys
import json
import httpx
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi.testclient import TestClient
from src.api.main import app
from openai import OpenAI


def _get_llm_client():
    """Returns (OpenAI client, model_name) for the configured LLM provider."""
    provider = (os.environ.get("LLM_PROVIDER") or "").lower()
    api_key = (
        os.environ.get("GROQ_API_KEY")
        or os.environ.get("GROK_API_KEY")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HF_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError("No LLM API key found. Set GROQ_API_KEY, HF_TOKEN, or OPENAI_API_KEY.")

    env_model = os.environ.get("LLM_MODEL", "").strip() or None

    if api_key.startswith("gsk_") or provider == "groq":
        base_url = "https://api.groq.com/openai/v1"
        model_name = env_model or "qwen/qwen3.6-27b"
    elif api_key.startswith("hf_") or provider in ["huggingface", "hf"]:
        base_url = "https://api-inference.huggingface.co/v1"
        model_name = (
            env_model
            if env_model and not env_model.startswith("openai/")
            else "Qwen/Qwen2.5-Coder-32B-Instruct"
        )
    elif os.environ.get("GROK_API_KEY") and not api_key.startswith("sk-"):
        base_url = "https://api.x.ai/v1"
        model_name = env_model or "grok-beta"
    else:
        base_url = None
        model_name = env_model or "gpt-4o-mini"

    return OpenAI(api_key=api_key, base_url=base_url), model_name


def get_high_risk_customer_profile() -> Dict[str, Any]:
    """Fetches a high risk customer profile directly from the local Limier API via TestClient."""
    try:
        payload = {"filters": {"min_risk_level": "high"}}
        print("Initializing internal API cache... (this takes ~60-70 seconds)")
        with TestClient(app) as client:
            response = client.post("/score", json=payload)
            response.raise_for_status()
            results = response.json()
            if not results:
                print("No high risk customers found in the dataset.")
                return None
            return results[0]
    except Exception as e:
        print(f"Error fetching data from API: {e}")
        return None


def generate_risk_narrative(customer_profile: Dict[str, Any]) -> str:
    """
    Generates a concise 2-3 sentence plain-English risk narrative for the Customer Drawer.
    Uses the configured LLM provider (Groq/HF/xAI/OpenAI).
    Falls back to a template-based narrative if the LLM is unavailable.
    """
    try:
        client, model_name = _get_llm_client()
    except ValueError:
        return _template_narrative(customer_profile)

    cust_id = customer_profile.get("customer_id", "Unknown")
    risk_level = customer_profile.get("risk_level", "low").upper()
    final_score = customer_profile.get("final_score", 0.0)
    rules = customer_profile.get("triggered_rules", [])
    features = customer_profile.get("top_features", [])

    rules_text = (
        "; ".join(r.get("rule", "") for r in rules[:3]) if rules else "None"
    )
    features_text = (
        ", ".join(
            f"{f.get('feature', '?')} (SHAP={f.get('shap_contribution', 0):.3f})"
            for f in features[:3]
        )
        if features
        else "No SHAP data"
    )

    prompt = (
        f"You are a Senior AML Compliance Officer at a bank. Write exactly 2-3 sentences "
        f"explaining in plain English why customer {cust_id} has been flagged as {risk_level} risk "
        f"(score: {final_score:.1f}/100). "
        f"Rules triggered: {rules_text}. "
        f"Top ML features driving the score: {features_text}. "
        f"Be concise, factual, and avoid jargon. Do not use bullet points."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a concise AML compliance analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return _template_narrative(customer_profile)


def _template_narrative(customer_profile: Dict[str, Any]) -> str:
    """Template-based fallback narrative when LLM is unavailable."""
    cust_id = customer_profile.get("customer_id", "Unknown")
    risk_level = customer_profile.get("risk_level", "low").upper()
    score = customer_profile.get("final_score", 0.0)
    rules = customer_profile.get("triggered_rules", [])

    if rules:
        rule_names = ", ".join(r.get("rule", "unknown") for r in rules[:2])
        return (
            f"Customer {cust_id} has been assigned a {risk_level} risk rating "
            f"(score: {score:.1f}/100) based on deterministic rule violations: {rule_names}. "
            f"The hybrid ML ensemble also elevated this customer's anomaly score above "
            f"the threshold, indicating unusual transaction patterns that warrant review."
        )
    return (
        f"Customer {cust_id} received a {risk_level} risk score of {score:.1f}/100. "
        f"The ML anomaly models detected statistically unusual transaction behaviour "
        f"that deviates from this customer's historical baseline. No hard rules were triggered, "
        f"but the ensemble score exceeds the review threshold."
    )


def generate_sar(customer_profile: Dict[str, Any]) -> str:
    """Generates a SAR using the configured LLM provider (Groq, HF, xAI, or OpenAI)."""
    client, model_name = _get_llm_client()

    cust_id = customer_profile.get("customer_id")
    risk_level = customer_profile.get("risk_level", "low").upper()
    ml_score = customer_profile.get("ml_contribution")
    rules = json.dumps(customer_profile.get("triggered_rules", []), indent=2)
    shap_features = json.dumps(customer_profile.get("top_features", []), indent=2)

    if risk_level == "HIGH":
        report_type = "formal Suspicious Activity Report (SAR)"
        instructions = """
1. Write a professional, concise 3-paragraph SAR.
2. Paragraph 1: Executive Summary (Customer ID, risk level, and primary reason).
3. Paragraph 2: Deterministic Evidence (explain triggered rules in plain English).
4. Paragraph 3: Machine Learning Nuance (explain SHAP features — e.g. why high spike_ratio
   or round amounts are mathematically suspicious).
5. Do not invent new transactions or data. Rely ONLY on the provided context.
"""
    else:
        report_type = f"Risk Assessment Report (Risk Level: {risk_level})"
        instructions = """
1. Write a professional, concise 3-paragraph Risk Assessment.
2. Paragraph 1: Executive Summary (Customer ID, risk level, general overview).
3. Paragraph 2: Deterministic Findings (why no severe rules were triggered, or minor rules found).
4. Paragraph 3: ML Context (explain SHAP values are generally normal, indicating low/medium risk).
5. Conclude this customer does not currently warrant a SAR. Do not invent new data.
"""

    prompt = f"""
You need to write a {report_type} for a customer evaluated by the Limier AI system.

Raw data from the ML and Rules engines:
- Customer ID: {cust_id}
- Overall Risk Level: {risk_level}
- Isolation Forest Anomaly Score (0-1): {ml_score}

Deterministic Rules Triggered:
{rules}

Top XGBoost ML Features (SHAP Values):
{shap_features}

Instructions:
{instructions}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a Senior AML (Anti-Money Laundering) Compliance Officer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("1. Fetching high-risk customer profile from backend...")
    profile = get_high_risk_customer_profile()

    if profile:
        print(f"-> Selected {profile['customer_id']} with Risk Level: {profile['risk_level']}")
        print("\n2. Generating SAR via configured LLM provider...")
        try:
            sar_report = generate_sar(profile)
            print("\n================= SUSPICIOUS ACTIVITY REPORT =================\n")
            print(sar_report)
            print("\n==============================================================")
        except Exception as e:
            print(f"Failed to generate SAR: {e}")
