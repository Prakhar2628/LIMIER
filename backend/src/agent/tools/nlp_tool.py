"""
NLP Tool for AI Agent investigation.
Analyzes transaction text narratives for a customer and returns sentiment, AML keywords, and typology insights.
"""
from typing import Dict, Any
from src.nlp.nlp_processor import analyze_text_narrative, generate_customer_memos

def analyze_customer_nlp(customer_id: str, is_high_risk: bool = False) -> Dict[str, Any]:
    """
    Retrieves and performs NLP analysis on transaction memos for customer_id.
    """
    memos = generate_customer_memos(customer_id, is_suspicious=is_high_risk)
    analysis = analyze_text_narrative(memos)
    analysis["customer_id"] = customer_id
    return analysis
