import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import streamlit as st

from src.data_generator import generate_anomaly_data
from src.metrics import compute_financial_cost, find_optimal_threshold
from src.models import train_anomaly_models

# --- Configuration de la page ---
st.set_page_config(
    page_title="AnomalyEngine",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AnomalyEngine — Détection & Pilotage Opérationnel")

# --- Sidebar : Domaine & Coûts ---
st.sidebar.header("⚙️ Configuration Métier")

domain_choice = st.sidebar.selectbox(
    "Domaine d'application",
    options=["fintech", "iot", "saas"],
    format_func=lambda x: {
        "fintech": "Fintech & E-Commerce (Fraude)",
        "iot": "Industrie IoT (Maintenance)",
        "saas": "SaaS Cloud (Consommation API)"
    }[x]
)

df, cost_config = generate_anomaly_data(domain=domain_choice)

st.sidebar.subheader("💰 Matrice de Coûts (€)")
fn_cost = st.sidebar.number_input("Coût Faux Négatif (Anomalie manquée)", value=float(cost_config["fn_cost"]), step=10.0)
fp_cost = st.sidebar.number_input("Coût Faux Positif (Fausse alerte)", value=float(cost_config["fp_cost"]), step=1.0)

# --- Entraînement des modèles ---
with st.spinner("Calcul des features et entraînement..."):
    results = train_anomaly_models(df)

X_train = results.get("X_train", results["X_test"]) # Récupération X_train si présent
X_test = results["X_test"]
y_train = results.get("y_train", results["y_test"])
y_test = results["y_test"]
xgb_probs = results["xgb_probs"]
iso_probs = results["iso_probs"]

best_pnl = find_optimal_threshold(y_test.values, xgb_probs, fn_cost, fp_cost)

# --- Navigation 4 Onglets ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Données & Feature Engineering",
    "🤖 2. Modélisation & Détection",
    "🔍 3. Explicabilité (SHAP)",
    "🎯 4. Dashboard Global & Alertes"
])

# ==================== ONGLET 1 : DATA & FEATURES ====================
with tab1:
    st.subheader(f"📁 Exploration & Construction des Échantillons — {cost_config['domain_label']}")
    
    # 1. Métriques Train / Test
    st.markdown("### 📐 Volumétrie & Répartition Train / Test")
    col_tr, col_te, col_ratio = st.columns(3)
    
    n_train = len(X_train)
    n_test = len(X_test)
    total_obs = n_train + n_test
    
    col_tr.metric("Échantillon Train", f"{n_train:,} obs.", f"Taux anomalie: {y_train.mean():.1%}")
    col_te.metric("Échantillon Test", f"{n_test:,} obs.", f"Taux anomalie: {y_test.mean():.1%}")
    col_ratio.metric("Ratio Split", f"{n_train/total_obs:.0%} Train / {n_test/total_obs:.0%} Test")

    st.markdown("---")

    # 2. Contextualisation des Variables selon le Domaine
    st.markdown("### 📝 Dictionnaire des Variables & Contextualisation Métier")
    
    # Explications adaptées dynamiquement au domaine
    domain_contexts = {
        "fintech": [
            "Montant moyen des transactions sur la fenêtre récente (€)",
            "Volatilité des montants payés (détecte les sauts soudains)",
            "Écart par rapport à l'historique habituel du porteur de carte",
            "Nombre total d'opérations exécutées dans la fenêtre d'analyse"
        ],
        "iot": [
            "Moyenne glissante des métriques capteurs (température, pression, vibration)",
            "Écart-type glissant (mesure la stabilité mécanique/thermique)",
            "Écart par rapport au régime nominal de la machine",
            "Nombre total de signaux/telemetries collectés sur la période"
        ],
        "saas": [
            "Débit moyen de requêtes API reçues par seconde (req/s)",
            "Variance de la latence ou du volume de requêtes",
            "Écart par rapport au quota/comportement habituel du compte client",
            "Volume total d'appels API enregistrés sur la fenêtre"
        ]
    }
    
    contexts = domain_contexts.get(domain_choice, domain_contexts["fintech"])
    adjusted_contexts = (contexts * ((len(results["feature_names"]) // len(contexts)) + 1))[:len(results["feature_names"])]

    features_desc = pd.DataFrame({
        "Variable": results["feature_names"],
        "Signification dans le contexte actuel": adjusted_contexts
    })
    st.dataframe(features_desc, use_container_width=True, hide_index=True)

    st.markdown("---")
    
    # 3. Distribution des variables
    st.markdown("### 📉 Distribution des Variables (Normal vs Anomalie)")
    feature_to_plot = st.selectbox("Sélectionner une variable à observer :", options=results["feature_names"])
    
    plot_df = X_test.copy()
    plot_df["Statut"] = y_test.map({0: "Normal", 1: "Anomalie"})
    
    fig_dist = px.histogram(
        plot_df, 
        x=feature_to_plot, 
        color="Statut", 
        barmode="overlay",
        title=f"Distribution de '{feature_to_plot}' sur l'échantillon Test",
        template="plotly_dark",
        color_discrete_map={"Normal": "#3B82F6", "Anomalie": "#EF4444"}
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ==================== ONGLET 2 : MODÉLISATION ====================
with tab2:
    st.subheader("⚙️ Performance des Modèles & Arbitrage Financier")
    st.markdown("""
    Comparaison entre une approche supervisée (**XGBoost**) et non-supervisée (**Isolation Forest**).
    Le seuil de décision est ajusté automatiquement pour minimiser le coût financier global.
    """)
    
    iso_pnl = find_optimal_threshold(y_test.values, iso_probs, fn_cost, fp_cost)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("### Benchmark des Stratégies")
        comp_df = pd.DataFrame({
            "Modèle": ["XGBoost (Cost-Sensitive)", "Isolation Forest (Non-Supervisé)"],
            "Seuil Optimal": [f"{best_pnl['threshold']:.2f}", f"{iso_pnl['threshold']:.2f}"],
            "Gain Net (€)": [f"{best_pnl['savings']:,.2f} €", f"{iso_pnl['savings']:,.2f} €"],
            "Anomalies Détectées (TP)": [f"{best_pnl['tp']} / {y_test.sum()}", f"{iso_pnl['tp']} / {y_test.sum()}"],
            "Fausses Alarmes (FP)": [best_pnl["fp"], iso_pnl["fp"]]
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with col_m2:
        st.markdown("### Arbitrage Coût vs Seuil (XGBoost)")
        thresholds = np.linspace(0.01, 0.99, 100)
        costs = [compute_financial_cost(y_test.values, xgb_probs, th, fn_cost, fp_cost)["total_cost"] for th in thresholds]
        
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(x=thresholds, y=costs, mode='lines', name='Coût Total (€)', line=dict(color='#EF4444', width=3)))
        fig_cost.add_vline(x=best_pnl['threshold'], line_dash="dash", line_color="#10B981", annotation_text=f"Seuil Optimal: {best_pnl['threshold']:.2f}")
        fig_cost.update_layout(title="Optimisation du Seuil de Risque", xaxis_title="Seuil de Probabilité", yaxis_title="Coût Financier (€)", template="plotly_dark", height=300)
        st.plotly_chart(fig_cost, use_container_width=True)

# ==================== ONGLET 3 : SHAP ====================
with tab3:
    st.subheader("🔍 Explicabilité Globale & Inspection d'Échantillons (SHAP)")
    shap_vals = results["shap_values"]

    # --- PARTIE 1 : DEUX GRAPHES GLOBAUX EN PREMIER ---
    st.markdown("### 1. Vue Globale : Importance & Impact des Features (Beeswarm & Importance)")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("#### Importance Globale Moyenne")
        mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
        shap_df = pd.DataFrame({"Feature": results["feature_names"], "Impact Moyen": mean_abs_shap}).sort_values(by="Impact Moyen", ascending=True)

        fig_shap_bar = px.bar(
            shap_df, 
            x="Impact Moyen", 
            y="Feature", 
            orientation='h', 
            template="plotly_dark", 
            color="Impact Moyen", 
            color_continuous_scale="Reds"
        )
        fig_shap_bar.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_shap_bar, use_container_width=True)

    with col_g2:
        st.markdown("#### SHAP Beeswarm Plot")
        fig_bee, ax_bee = plt.subplots(figsize=(7, 4.5))
        shap.plots.beeswarm(shap_vals, show=False, max_display=8)
        plt.tight_layout()
        st.pyplot(fig_bee)
        plt.close(fig_bee)

    st.markdown("---")

    # --- PARTIE 2 : COMPARAISON CAS NORMAL VS ANOMALIE ---
    st.markdown("### 2. Inspection Comparative : Cas Normal vs Cas d'Anomalie")
    st.caption("Sélectionnez un cas normal et un cas anormal pour observer la différence d'explication par le modèle.")

    # Isolation des index réels
    normal_indices = y_test[y_test == 0].index.tolist()
    anomaly_indices = y_test[y_test == 1].index.tolist()

    col_norm, col_anom = st.columns(2)

    with col_norm:
        st.markdown("#### 🟢 Exemple : Cas Normal (Sain)")
        selected_normal = st.selectbox("Échantillon Normal :", options=normal_indices[:50], index=0)
        
        idx_n = list(X_test.index).index(selected_normal)
        prob_n = xgb_probs[idx_n]
        
        st.write(f"**Score de Risque :** `{prob_n:.1%}` | **Décision :** ✅ Normal")

        fig_n, ax_n = plt.subplots(figsize=(6, 4))
        shap.plots.waterfall(shap_vals[idx_n], show=False, max_display=6)
        plt.tight_layout()
        st.pyplot(fig_n)
        plt.close(fig_n)

    with col_anom:
        st.markdown("#### 🔴 Exemple : Cas d'Anomalie (Alerte)")
        selected_anomaly = st.selectbox("Échantillon Anomalie :", options=anomaly_indices[:50], index=0)
        
        idx_a = list(X_test.index).index(selected_anomaly)
        prob_a = xgb_probs[idx_a]
        
        st.write(f"**Score de Risque :** `{prob_a:.1%}` | **Décision :** ⚠️ ALERTE")

        fig_a, ax_a = plt.subplots(figsize=(6, 4))
        shap.plots.waterfall(shap_vals[idx_a], show=False, max_display=6)
        plt.tight_layout()
        st.pyplot(fig_a)
        plt.close(fig_a)

# ==================== ONGLET 4 : DASHBOARD GLOBAL & PILOTAGE ====================
with tab4:
    st.subheader("🎯 Centre de Contrôle & File d'Attente des Alertes")
    st.markdown("Vue synthétique dédiée aux **équipes opérationnelles** pour traiter prioritairement les alertes détectées.")

    test_results = X_test.copy()
    test_results["Score_Risque"] = xgb_probs
    test_results["Statut_Predit"] = np.where(test_results["Score_Risque"] >= best_pnl["threshold"], "⚠️ ALERTE", "✅ OK")
    test_results["Statut_Reel"] = y_test.map({0: "Normal", 1: "Anomalie"})
    
    c1, c2, c3, c4 = st.columns(4)
    alert_count = (test_results["Statut_Predit"] == "⚠️ ALERTE").sum()
    c1.metric("Total Échantillons", len(test_results))
    c2.metric("Alertes Générées", alert_count)
    c3.metric("Taux de Détection", f"{(best_pnl['tp'] / y_test.sum()):.1%}")
    c4.metric("Économies Totales", f"{best_pnl['savings']:,.2f} €")

    st.markdown("---")
    st.markdown("### File de Traitement des Alertes Critiques")
    
    alerts_df = test_results[test_results["Statut_Predit"] == "⚠️ ALERTE"].sort_values(by="Score_Risque", ascending=False)
    
    st.dataframe(
        alerts_df,
        column_config={
            "Score_Risque": st.column_config.ProgressColumn(
                "Niveau de Risque",
                format="%.2f",
                min_value=0,
                max_value=1
            )
        },
        use_container_width=True
    )