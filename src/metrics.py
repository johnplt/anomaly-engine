import numpy as np
from sklearn.metrics import confusion_matrix


def compute_financial_cost(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float, fn_cost: float, fp_cost: float) -> dict:
    """Calcule la matrice de confusion et le coût financier global pour un seuil donné."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_cost = (fn * fn_cost) + (fp * fp_cost)
    
    # Coût naïf (si aucune détection : toutes les anomalies sont des FN)
    naive_cost = (y_true.sum()) * fn_cost
    savings = naive_cost - total_cost

    return {
        "threshold": threshold,
        "total_cost": total_cost,
        "savings": savings,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


def find_optimal_threshold(y_true: np.ndarray, y_pred_prob: np.ndarray, fn_cost: float, fp_cost: float) -> dict:
    """Parcourt les seuils de probabilité pour trouver celui qui minimise le coût financier."""
    thresholds = np.linspace(0.01, 0.99, 100)
    best_res = None
    min_cost = float("inf")

    for th in thresholds:
        res = compute_financial_cost(y_true, y_pred_prob, th, fn_cost, fp_cost)
        if res["total_cost"] < min_cost:
            min_cost = res["total_cost"]
            best_res = res

    return best_res