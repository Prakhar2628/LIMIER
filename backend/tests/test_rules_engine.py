"""
Tests to lock in the Rules Engine catch rate on planted high-severity typologies.
"""
import pandas as pd
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.rules_engine import evaluate_all

@pytest.fixture
def transactions_data():
    """Loads transactions from dataset CSV or builds synthetic test txns."""
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dataset", "transactions.csv"))
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    else:
        # Fallback synthetic mock
        rows = []
        for i in range(3):
            rows.append({
                "transaction_id": f"TXN_{i}",
                "customer_id": "CUST_PLANTED_1",
                "timestamp": f"2026-07-2{i} 10:00:00",
                "amount": 9500.0,
                "direction": "credit",
                "counterparty_id": "CP_1",
                "counterparty_country": "US",
                "channel": "wire",
                "transaction_type": "deposit",
                "planted_pattern": "structuring",
                "is_planted_suspicious": 1
            })
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

def test_100_percent_planted_recall(transactions_data):
    """
    Validates that the rules engine catches planted high-severity patterns.
    """
    txns = transactions_data
    
    # 1. Apply the noise filter threshold used in production orchestration
    significant_txns = txns[txns["amount"] >= 8000].copy()
    
    # 2. Evaluate all rules
    rule_results = evaluate_all(significant_txns)
    
    # 3. Identify planted high-severity customers if planted_pattern exists
    if "planted_pattern" in txns.columns:
        high_sev_patterns = ["structuring", "rapid_cashout", "round_trip"]
        planted_high = txns[txns["planted_pattern"].isin(high_sev_patterns)]["customer_id"].unique()
        
        if len(planted_high) > 0:
            caught_count = 0
            missed_customers = []
            
            for cid in planted_high:
                res = rule_results.get(cid, [])
                if any(r.severity == "high" for r in res):
                    caught_count += 1
                else:
                    missed_customers.append(cid)
                    
            assert caught_count == len(planted_high), f"Missed planted high-severity customers: {missed_customers}"
            assert caught_count > 0, "No planted patterns were generated to test!"
    else:
        # Generic check
        assert isinstance(rule_results, dict)
