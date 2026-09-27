import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Génère les variables d'écart (Z-scores et rolling stats) valables pour tous les domaines."""
    data = df.copy()
    
    # 1. Z-score sur la métrique principale
    mean_val = data["metric_value"].mean()
    std_val = data["metric_value"].std() + 1e-6
    data["z_score_metric"] = (data["metric_value"] - mean_val) / std_val

    # 2. Moyennes glissantes sur 5 périodes
    data["rolling_mean_5"] = data["metric_value"].rolling(window=5, min_periods=1).mean()
    data["ratio_to_rolling_mean"] = data["metric_value"] / (data["rolling_mean_5"] + 1e-6)

    # 3. Delta par rapport au point précédent
    data["delta_prev"] = data["metric_value"].diff().fillna(0)

    features = ["metric_value", "frequency_score", "z_score_metric", "ratio_to_rolling_mean", "delta_prev"]
    return data[features]


def train_anomaly_models(df: pd.DataFrame) -> dict:
    """Entraîne Isolation Forest et XGBoost sur les données préparées."""
    X = prepare_features(df)
    y = df["is_anomaly"]

    # Split temporel/stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 1. Isolation Forest (Non-supervisé) ---
    iso_forest = IsolationForest(
        contamination=float(y.mean()), random_state=42, n_jobs=-1
    )
    iso_forest.fit(X_train)
    
    # Calcul des scores d'anomalie normalisés entre 0 et 1
    raw_scores = -iso_forest.score_samples(X_test)
    iso_probs = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-6)

    # --- 2. XGBoost Cost-Sensitive (Supervisé) ---
    # Calcul du poids pour gérer le fort déséquilibre de classes
    scale_pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-6)
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    # --- 3. Explicabilité SHAP pour XGBoost ---
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "iso_probs": iso_probs,
        "xgb_probs": xgb_probs,
        "xgb_model": xgb_model,
        "shap_values": shap_values,
        "feature_names": X.columns.tolist()
    }