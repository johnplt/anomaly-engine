# AnomalyEngine — Détection d'anomalies et arbitrage financier

**AnomalyEngine** est une plateforme décisionnelle multi-domaines (Fintech, IoT, SaaS), un modèle de détection d'anomalies orienté sur l'optimisation financière du seuil de décision (arbitrage fausses alertes vs anomalies manquées).

L'application arbitre le seuil de décision pour minimiser le coût global d'exploitation (**Faux Négatifs vs Faux Positifs**) et rend chaque alerte actionnable grâce aux explications **SHAP**.

---

## 🎯 Valeur Métier & Fonctionnalités

- **Feature Engineering & Prétraitement** : Calcul de métriques glissantes, de volatilité et d'écarts aux comportements nominaux, adaptées à la détection de ruptures de comportement.
- **Modélisation & Benchmarking** : Alignement du seuil de probabilité sur le coût réel d'un Faux Négatif (client perdu) vs Faux Positif (campagne inutile).
- **Explicabilité globale & locale** : Analyse fine des facteurs déclencheurs d'alertes via SHAP values (Beeswarm plot global et Waterfall plots comparatifs sur cas sains vs anormaux).
- **Arbitrage coût/seuil** : Simulation interactive recherchant le seuil de probabilité optimal minimisant la matrice de coûts métier pour maximiser le gain net.

---

## 🛠️ Architecture du Projet

```text
├── .github/
│   └── workflows/
│       └── ci.yml             # Pipeline CI (Linter & Tests)
├── src/
│   ├── data_generator.py      # Génération/injection de données multi-domaines
│   ├── metrics.py             # Logique d'optimisation coût & P&L
│   └── models.py              # Pipelines d'entraînement (XGBoost & Isolation Forest)
├── app.py                     # Application Streamlit (4 onglets)
├── requirements.txt           # Dépendances Python
└── README.md
```

## Installation & Lancement

```bash
# 1. Cloner le dépôt
git clone [https://github.com/votre-user/anomaly-engine.git](https://github.com/votre-user/anomaly-engine.git)
cd anomaly-engine

# 2. Créer un environnement virtuel et installer les dépendances
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Lancer l'application Streamlit
streamlit run app.py
```