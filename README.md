# AnomalyEngine — Détection d'Anomalies & Optimisation P&L

**AnomalyEngine** est une plateforme décisionnelle multi-domaines (Fintech, IoT, SaaS) qui transforme les scores de risque ML en **décisions financières optimales**.

L'application arbitre le seuil de décision pour minimiser le coût global d'exploitation (**Faux Négatifs vs Faux Positifs**) et rend chaque alerte actionnable grâce aux explications **SHAP**.

---

## 🎯 Valeur Métier & Fonctionnalités

- **Optimisation Financière (P&L) :** Détermination automatique du seuil de décision optimal selon une matrice de coûts personnalisée (`€/FN` vs `€/FP`).
- **Benchmark de Modèles :** Comparaison entre une approche supervisée (*XGBoost*) et non supervisée (*Isolation Forest*).
- **Explicabilité Avancée (SHAP) :** Analyse globale (Bar plot & Beeswarm) et explications locales par échantillon (Waterfall plots pour cas sains vs anormaux).
- **Console Opérationnelle :** File d'attente dynamique triée par niveau de risque pour les équipes métier.

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