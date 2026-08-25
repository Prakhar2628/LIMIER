import os
import sys
import time
import logging
import pandas as pd
import joblib

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.tools.anomaly_detection_tool import detect_anomalies
from src.agent.tools.risk_classification_tool import classify_risk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainAndCache")

def train_and_cache():
    logger.info("Starting model training & pre-computation workflow...")
    start_t = time.time()
    
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(backend_dir, "data", "dataset")
    cache_dir = os.path.join(backend_dir, "data", "cache")
    models_dir = os.path.join(backend_dir, "models")
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    logger.info(f"Loading raw datasets from {data_dir}...")
    txns = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
    custs = pd.read_csv(os.path.join(data_dir, "customers.csv"))
    
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    logger.info(f"Loaded {len(txns)} transactions and {len(custs)} customers.")
    
    logger.info("Building AML features and fitting ML anomaly models (IsolationForest, XGBoost, MetaEnsemble)...")
    # Force re-fit models by deleting old pkls if present to ensure clean training
    for model_name in ["iso_scorer.pkl", "xgb_scorer.pkl", "meta_ensemble.pkl"]:
        p = os.path.join(models_dir, model_name)
        if os.path.exists(p):
            os.remove(p)
            
    features_df, iso_scorer, xgb_scorer, meta_ensemble = detect_anomalies(txns)
    
    logger.info("Classifying hybrid risk levels and generating SHAP explanations...")
    risk_summary_df = classify_risk(txns, features_df)
    
    # Save pre-computed cache files for instant startup (< 1 second)
    risk_cache_path = os.path.join(cache_dir, "risk_summary_df.pkl")
    features_cache_path = os.path.join(cache_dir, "features_df.pkl")
    
    logger.info(f"Saving pre-scored risk summary cache to {risk_cache_path}...")
    joblib.dump(risk_summary_df, risk_cache_path)
    joblib.dump(features_df, features_cache_path)
    
    total_time = time.time() - start_t
    logger.info(f"SUCCESS: Training and caching finished in {total_time:.2f} seconds.")
    logger.info("Models saved in 'backend/models/' and pre-scored risk cache saved in 'backend/data/cache/'.")

if __name__ == "__main__":
    train_and_cache()
