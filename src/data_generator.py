import numpy as np
import pandas as pd


def generate_anomaly_data(domain: str = "fintech", n_samples: int = 5000, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Génère un dataset d'anomalies synthétiques et renvoie la matrice de coûts par défaut."""
    np.random.seed(seed)
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq="h")

    if domain == "fintech":
        # Transaction Amount + Velocity
        amount = np.random.exponential(scale=50, size=n_samples) + 10
        velocity = np.random.poisson(lam=2, size=n_samples)
        
        # Anomalies : gros montants ou forte vélocité
        is_anomaly = (np.random.rand(n_samples) < 0.015)
        amount[is_anomaly] *= np.random.uniform(5, 15, size=is_anomaly.sum())
        velocity[is_anomaly] += np.random.randint(5, 12, size=is_anomaly.sum())

        df = pd.DataFrame({
            "timestamp": timestamps,
            "metric_value": amount,
            "frequency_score": velocity,
            "is_anomaly": is_anomaly.astype(int)
        })
        
        cost_config = {
            "fn_cost": 120.0,  # Coût d'une fraude manquée (Chargeback + pénalité)
            "fp_cost": 5.0,    # Coût d'une fausse alerte (Revue manuelle support)
            "unit_name": "Transaction (€)",
            "domain_label": "Fintech & E-Commerce (Fraude)"
        }

    elif domain == "iot":
        # Capteur Industriel : Température + Vibration
        temp = np.random.normal(loc=65, scale=5, size=n_samples)
        vibration = np.random.normal(loc=1.2, scale=0.2, size=n_samples)
        
        is_anomaly = (np.random.rand(n_samples) < 0.02)
        temp[is_anomaly] += np.random.uniform(15, 35, size=is_anomaly.sum())
        vibration[is_anomaly] *= np.random.uniform(2, 4, size=is_anomaly.sum())

        df = pd.DataFrame({
            "timestamp": timestamps,
            "metric_value": temp,
            "frequency_score": vibration,
            "is_anomaly": is_anomaly.astype(int)
        })

        cost_config = {
            "fn_cost": 800.0,  # Coût d'une casse machine / arrêt de chaîne
            "fp_cost": 40.0,   # Coût de déplacement inutile d'un technicien
            "unit_name": "Température (°C)",
            "domain_label": "Industrie IoT (Maintenance Prédictive)"
        }

    else:  # saas
        # Consommation API
        api_calls = np.random.negative_binomial(n=5, p=0.1, size=n_samples)
        error_rate = np.random.beta(a=1, b=20, size=n_samples)

        is_anomaly = (np.random.rand(n_samples) < 0.018)
        api_calls[is_anomaly] *= np.random.randint(4, 10, size=is_anomaly.sum())
        error_rate[is_anomaly] = np.random.uniform(0.3, 0.8, size=is_anomaly.sum())

        df = pd.DataFrame({
            "timestamp": timestamps,
            "metric_value": api_calls,
            "frequency_score": error_rate,
            "is_anomaly": is_anomaly.astype(int)
        })

        cost_config = {
            "fn_cost": 300.0,  # Coût fuite de crédits / Attaque DDOS non stoppée
            "fp_cost": 15.0,   # Coût de re-validation / alerte Slack dev
            "unit_name": "Requêtes / min",
            "domain_label": "SaaS Cloud (Anomalies de Consommation)"
        }

    return df, cost_config