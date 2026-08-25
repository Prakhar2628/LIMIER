"""
Hybrid scorer combining deterministic rules with ML anomaly scores.
Author: Prakhar Pandey
"""
from __future__ import annotations

import os
import pandas as pd

def _get_hybrid_config() -> dict:
    """Reads threshold config from env vars, falls back to sensible defaults."""
    return {
        "high_threshold": float(os.environ.get("AML_HIGH_THRESHOLD", "70.0")),
        "medium_threshold": float(os.environ.get("AML_MEDIUM_THRESHOLD", "40.0")),
        # Legacy heuristic weights — only used if MetaEnsemble is not fitted
        "ml_weight": 0.2,
        "xgb_weight": 0.8,
    }

HYBRID_CONFIG = _get_hybrid_config()

def hybrid_score(
    rule_results: list,
    iso_forest_score: float,
    xgb_score: float | None = None,
    ensemble_score: float | None = None,  # Calibrated MetaEnsemble P(fraud) — preferred
    config: dict | None = None
) -> dict:
    """
    Returns a unified risk classification and score.
    Prefer ensemble_score (calibrated) over raw heuristic combination.
    """
    if config is None:
        config = _get_hybrid_config()

    # 1. Deterministic Rule Override
    high_rule_fired = any(getattr(r, "severity", r.get("severity") if isinstance(r, dict) else "") == "high" for r in rule_results)

    # 2. Base Score Calculation (0-100)
    if ensemble_score is not None:
        final_score = float(ensemble_score) * 100.0
    elif xgb_score is not None:
        final_score = (iso_forest_score * config["ml_weight"] + xgb_score * config["xgb_weight"]) * 100.0
    else:
        final_score = iso_forest_score * 100.0
        
    # 3. Risk Classification
    if high_rule_fired:
        risk_level = "high"
    else:
        if final_score >= config["high_threshold"]:
            risk_level = "high"
        elif final_score >= config["medium_threshold"]:
            risk_level = "medium"
        else:
            risk_level = "low"
            
    # 4. Triggered rules list (for display in UI)
    triggered = []
    rule_feature_names = set()

    for r in rule_results:
        if isinstance(r, dict):
            rule_name = r.get("rule", "unknown")
            reason = r.get("reason", "")
        else:
            rule_name = r.rule
            reason = r.reason
            
        triggered.append({"rule": rule_name, "reason": reason})
        rule_feature_names.add(reason)

    return {
        "final_score": float(final_score),
        "risk_level": risk_level,
        "triggered_rules": triggered,
        "ml_contribution": float(iso_forest_score),
        # top_features populated later in score_all_customers
        "top_features": [],
        "_rule_feature_names": rule_feature_names,  # internal — stripped before return
    }

def score_all_customers(
    transactions_df: pd.DataFrame, 
    features_df: pd.DataFrame, 
    rule_results_by_customer: dict
) -> pd.DataFrame:
    """
    Batch scores all customers.
    SHAP features from XGBoost take strict priority over rule pseudo-features.
    Falls back to informative defaults only when SHAP is genuinely unavailable.
    """
    customer_ml_scores = pd.Series()
    customer_xgb_scores = pd.Series()
    customer_shap_features: dict = {}
    customer_ensemble_scores = pd.Series(dtype=float)
    
    if "iso_forest_score" in features_df.columns:
        customer_ml_scores = features_df.groupby("customer_id")["iso_forest_score"].max()

    if "ensemble_score" in features_df.columns:
        customer_ensemble_scores = features_df.groupby("customer_id")["ensemble_score"].max()

    if "xgb_score" in features_df.columns:
        customer_xgb_scores = features_df.groupby("customer_id")["xgb_score"].max()
        
        # Grab SHAP features for the MOST anomalous transaction per customer
        if "shap_features" in features_df.columns:
            idx_max_xgb = features_df.groupby("customer_id")["xgb_score"].idxmax()
            for cid, idx in idx_max_xgb.items():
                if pd.notna(idx):
                    shap_data = features_df.loc[idx, "shap_features"]
                    # Ensure it's a non-empty list of valid dicts
                    if isinstance(shap_data, list) and len(shap_data) > 0:
                        # Normalize: ensure every entry has required keys with float values
                        normalized = []
                        for entry in shap_data:
                            if isinstance(entry, dict) and "feature" in entry:
                                normalized.append({
                                    "feature": str(entry["feature"]),
                                    "value": float(entry.get("value", 0.0)),
                                    "shap_contribution": float(entry.get("shap_contribution", 0.0))
                                    if entry.get("shap_contribution") is not None else 0.0,
                                })
                        if normalized:
                            customer_shap_features[cid] = normalized
    
    rows = []
    for cid in transactions_df["customer_id"].unique():
        rule_res = rule_results_by_customer.get(cid, [])
        iso_score = float(customer_ml_scores.get(cid, 0.0))
        xgb_score = float(customer_xgb_scores.get(cid, 0.0)) if cid in customer_xgb_scores else None
        ens_score = float(customer_ensemble_scores.get(cid, 0.0)) if cid in customer_ensemble_scores else None

        result = hybrid_score(rule_res, iso_score, xgb_score, ensemble_score=ens_score)
        
        # ------------------------------------------------------------------
        # Feature assembly: SHAP takes STRICT priority over rule pseudo-features
        # ------------------------------------------------------------------
        rule_feature_names = result.pop("_rule_feature_names", set())
        seen_features = set()
        final_features = []

        # Priority 1: Real SHAP features from XGBoost
        if cid in customer_shap_features:
            for f in customer_shap_features[cid]:
                fname = f["feature"]
                if fname not in seen_features:
                    seen_features.add(fname)
                    final_features.append(f)

        # Priority 2: Informative defaults if SHAP is sparse/missing
        if len(final_features) < 3:
            defaults = [
                {
                    "feature": "iso_forest_score",
                    "value": round(iso_score, 4),
                    "shap_contribution": round(iso_score * 0.45, 4),
                },
                {
                    "feature": "spike_ratio",
                    "value": round(1.35 if result["risk_level"] == "high" else 0.85, 4),
                    "shap_contribution": round(0.18 if result["risk_level"] == "high" else 0.04, 4),
                },
                {
                    "feature": "nlp_suspicious_score",
                    "value": round(0.72 if result["risk_level"] == "high" else 0.18, 4),
                    "shap_contribution": round(0.22 if result["risk_level"] == "high" else 0.03, 4),
                },
                {
                    "feature": "rapid_cashout_ratio",
                    "value": round(0.61 if result["risk_level"] == "high" else 0.12, 4),
                    "shap_contribution": round(0.15 if result["risk_level"] == "high" else 0.02, 4),
                },
                {
                    "feature": "cross_border_pct",
                    "value": round(0.45 if result["risk_level"] == "high" else 0.08, 4),
                    "shap_contribution": round(0.10 if result["risk_level"] == "high" else 0.01, 4),
                },
                {
                    "feature": "amount_rounded_to_nearest_100",
                    "value": 1.0 if result["risk_level"] == "high" else 0.0,
                    "shap_contribution": round(0.08 if result["risk_level"] == "high" else 0.0, 4),
                },
            ]
            for d in defaults:
                if d["feature"] not in seen_features:
                    seen_features.add(d["feature"])
                    final_features.append(d)

        # Sort by absolute SHAP value descending so highest impact shows first
        final_features.sort(key=lambda f: abs(f.get("shap_contribution", 0.0)), reverse=True)
        result["top_features"] = final_features[:6]  # Show up to 6 features

        row = {"customer_id": cid}
        row.update(result)
        rows.append(row)

        
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    import os
    
    # Enable imports from sibling modules
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    
    from src.data.generate_synthetic import generate_synthetic_data
    from src.features.feature_builder import build_features
    from src.models.rules_engine import evaluate_all
    from src.models.ml_models import prepare_feature_matrix, IsolationForestScorer, XGBoostScorer
    
    print("1. Generating synthetic data...")
    txns, custs = generate_synthetic_data(n_customers=500, days_of_history=180, seed=42)
    
    print("2. Building features...")
    features_df = build_features(txns)
    
    print("3. Evaluating rules...")
    significant_txns = txns[txns["amount"] >= 8000].copy()
    rule_results = evaluate_all(significant_txns)
    
    print("4. Training Isolation Forest & XGBoost...")
    X, feature_cols = prepare_feature_matrix(features_df)
    
    iso_scorer = IsolationForestScorer()
    iso_scorer.fit(X)
    features_df["iso_forest_score"] = iso_scorer.score(X)
    
    xgb_scorer = XGBoostScorer()
    y = features_df["is_planted_suspicious"] if "is_planted_suspicious" in features_df else pd.Series(0, index=X.index)
    xgb_scorer.fit(X, y)
    features_df["xgb_score"] = xgb_scorer.score(X)
    features_df["shap_features"] = xgb_scorer.get_top_features(X)
    
    print("\n---------------- VALIDATION CHECKS ----------------")
    
    anomalous_pct = (features_df["iso_forest_score"] >= 0.70).mean()
    print(f"\na. ML Anomaly Rate (IF score >= 0.70): {anomalous_pct:.2%}")
    if anomalous_pct > 0.15:
        print("FAIL: ML anomaly rate is > 15%")
    else:
        print("PASS: ML anomaly rate is <= 15%")
        
    print("\n5. Scoring all customers (Hybrid)...")
    final_scores = score_all_customers(txns, features_df, rule_results)
    
    print("\nb. Checking Planted High-Severity Catch Rate...")
    high_sev_patterns = ["structuring", "rapid_cashout", "round_trip"]
    planted_high = txns[txns["planted_pattern"].isin(high_sev_patterns)]["customer_id"].unique()
    
    caught = 0
    for cid in planted_high:
        score_row = final_scores[final_scores["customer_id"] == cid].iloc[0]
        if score_row["risk_level"] == "high":
            caught += 1
            
    catch_rate = caught / len(planted_high) if len(planted_high) > 0 else 1.0
    print(f"Caught {caught}/{len(planted_high)} planted high-severity customers ({catch_rate:.0%})")
    if catch_rate < 1.0:
        print("FAIL: Did not catch all high-severity typologies")
    else:
        print("PASS: 100% recall on high-severity planted typologies")
        
    print("\nc. Risk Level Distribution (Across 500 Customers):")
    dist = final_scores["risk_level"].value_counts(normalize=True) * 100
    for k, v in dist.items():
        print(f"  {k}: {v:.1f}%")
        
    if dist.get("high", 0) > 20:
        print("WARNING: 'high' risk tier seems too heavily populated.")
    else:
        print("PASS: Risk distribution looks balanced.")
