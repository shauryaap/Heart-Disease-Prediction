import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction — ML Analytics App",
    page_icon="❤️",
    layout="wide",
)

st.title("❤️ Heart Disease Prediction Using Machine Learning")
st.caption(
    "Kaggle Playground Series S6E2 (630,000 rows, 13 predictors) | Academic & Educational Use Only"
)

# -----------------------------------------------------------------------------
# LOAD ARTIFACTS (FROM COLAB) IF AVAILABLE
# -----------------------------------------------------------------------------
MODEL_PATH = os.path.join("artifacts", "best_heart_disease_model.joblib")


@st.cache_resource
def load_saved_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


best_model = load_saved_model()

# Verified benchmark results from the Colab notebook run
FULL_SPLIT_RESULTS = pd.DataFrame(
    [
        {"Model": "HistGradientBoosting (Boosting)", "Accuracy": 0.8893, "Precision": 0.8823, "Recall": 0.8692, "F1": 0.8757, "ROC_AUC": 0.9557, "CV_AUC": 0.9533},
        {"Model": "Random Forest", "Accuracy": 0.8858, "Precision": 0.8675, "Recall": 0.8797, "F1": 0.8736, "ROC_AUC": 0.9530, "CV_AUC": 0.9511},
        {"Model": "Logistic Regression (L2)", "Accuracy": 0.8871, "Precision": 0.8838, "Recall": 0.8613, "F1": 0.8724, "ROC_AUC": 0.9537, "CV_AUC": 0.9539},
        {"Model": "Logistic Regression (L1)", "Accuracy": 0.8870, "Precision": 0.8838, "Recall": 0.8611, "F1": 0.8723, "ROC_AUC": 0.9537, "CV_AUC": 0.9539},
        {"Model": "Bagging", "Accuracy": 0.8858, "Precision": 0.8793, "Recall": 0.8639, "F1": 0.8716, "ROC_AUC": 0.9528, "CV_AUC": 0.9493},
        {"Model": "AdaBoost (Boosting)", "Accuracy": 0.8861, "Precision": 0.8821, "Recall": 0.8610, "F1": 0.8714, "ROC_AUC": 0.9533, "CV_AUC": 0.9525},
        {"Model": "Decision Tree", "Accuracy": 0.8769, "Precision": 0.8568, "Recall": 0.8711, "F1": 0.8639, "ROC_AUC": 0.9393, "CV_AUC": 0.9244},
    ]
)

SAMPLED_RESULTS = pd.DataFrame(
    [
        {"Model": "Linear SVM (30k/15k)", "Accuracy": 0.8852, "Precision": 0.8812, "Recall": 0.8619, "F1": 0.8714, "ROC_AUC": 0.9547},
        {"Model": "KNN (30k/15k)", "Accuracy": 0.8697, "Precision": 0.8599, "Recall": 0.8498, "F1": 0.8548, "ROC_AUC": 0.9315},
        {"Model": "SVM Linear kernel (8k/4k)", "Accuracy": 0.8813, "Precision": 0.8774, "Recall": 0.8544, "F1": 0.8658, "ROC_AUC": 0.9481},
        {"Model": "SVM Exponential kernel (8k/4k)", "Accuracy": 0.8770, "Precision": 0.8702, "Recall": 0.8528, "F1": 0.8614, "ROC_AUC": 0.9469},
        {"Model": "SVM RBF kernel (8k/4k)", "Accuracy": 0.8775, "Precision": 0.8712, "Recall": 0.8528, "F1": 0.8619, "ROC_AUC": 0.9436},
        {"Model": "SVM Polynomial kernel (8k/4k)", "Accuracy": 0.8743, "Precision": 0.8694, "Recall": 0.8466, "F1": 0.8579, "ROC_AUC": 0.9394},
        {"Model": "SVM Sigmoid kernel (8k/4k)", "Accuracy": 0.8158, "Precision": 0.7980, "Recall": 0.7886, "F1": 0.7933, "ROC_AUC": 0.8934},
    ]
)

REGRESSION_RESULTS = pd.DataFrame(
    [
        {"Method": "Simple Linear Regression (Max HR)", "Predictors": 1, "R2": 0.1968, "RMSE": 0.4457, "Accuracy_0.5": 0.7010, "F1": 0.6234},
        {"Method": "Multiple Linear Regression", "Predictors": "all", "R2": 0.6173, "RMSE": 0.3077, "Accuracy_0.5": 0.8843, "F1": 0.8678},
        {"Method": "Lasso Regression (L1)", "Predictors": "all", "R2": 0.6103, "RMSE": 0.3105, "Accuracy_0.5": 0.8813, "F1": 0.8640},
        {"Method": "Ridge Regression (L2)", "Predictors": "all", "R2": 0.6173, "RMSE": 0.3077, "Accuracy_0.5": 0.8843, "F1": 0.8678},
    ]
)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "🩺 1. Live Risk Predictor",
        "📊 2. Model Comparison (13 Methods)",
        "🧬 3. PCA & K-Means Insights",
    ]
)

with tab1:
    st.subheader("Patient Clinical Feature Input (13 Predictors)")
    if best_model is None:
        st.info(
            "Using coefficient-based fallback until `artifacts/best_heart_disease_model.joblib` is uploaded to GitHub."
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 25, 85, 58)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
        chest_pain = st.selectbox("Chest pain type", [1, 2, 3, 4], index=3)
        bp = st.number_input("BP (mm Hg)", 80, 220, 140)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 239)
    with c2:
        fbs = st.selectbox("FBS over 120", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        ekg = st.selectbox("EKG results", [0, 1, 2], index=0)
        max_hr = st.slider("Max HR", 70, 210, 150)
        ex_angina = st.selectbox("Exercise angina", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    with c3:
        st_dep = st.slider("ST depression", 0.0, 6.5, 1.0, step=0.1)
        slope_st = st.selectbox("Slope of ST", [1, 2, 3], index=1)
        vessels = st.selectbox("Number of vessels fluro", [0, 1, 2, 3], index=0)
        thallium = st.selectbox("Thallium", [3, 6, 7], index=2)

    input_df = pd.DataFrame(
        [
            {
                "Age": age,
                "Sex": sex,
                "Chest pain type": chest_pain,
                "BP": bp,
                "Cholesterol": chol,
                "FBS over 120": fbs,
                "EKG results": ekg,
                "Max HR": max_hr,
                "Exercise angina": ex_angina,
                "ST depression": st_dep,
                "Slope of ST": slope_st,
                "Number of vessels fluro": vessels,
                "Thallium": thallium,
            }
        ]
    )

    if st.button("Predict Heart Disease Risk", type="primary"):
        if best_model is not None:
            prob = float(best_model.predict_proba(input_df)[0][1])
            pred = int(best_model.predict(input_df)[0])
        else:
            # Lightweight interpretable fallback aligned with L1 Logistic top features
            z = (
                -1.8
                + 0.03 * (age - 54)
                + 0.65 * sex
                + (1.4 if chest_pain == 4 else -0.4)
                + 0.01 * (bp - 130)
                - 0.025 * (max_hr - 150)
                + 0.85 * ex_angina
                + 0.45 * st_dep
                + 0.70 * vessels
                + (1.1 if thallium == 7 else -0.6)
            )
            prob = float(1.0 / (1.0 + np.exp(-z)))
            pred = 1 if prob >= 0.5 else 0

        m1, m2 = st.columns(2)
        m1.metric("Predicted Class", "Presence (1)" if pred == 1 else "Absence (0)")
        m2.metric("Presence Probability", f"{prob*100:.1f}%")

with tab2:
    st.subheader("Full-Split Classification & Ensemble Models (504k Train / 126k Test)")
    st.dataframe(FULL_SPLIT_RESULTS, use_container_width=True, hide_index=True)

    fig_auc = px.bar(
        FULL_SPLIT_RESULTS.sort_values("CV_AUC"),
        x="CV_AUC",
        y="Model",
        orientation="h",
        text="CV_AUC",
        title="3-Fold Cross-Validated ROC-AUC (Best: Logistic Regression L1 = 0.9539)",
    )
    fig_auc.update_xaxes(range=[0.90, 0.97])
    st.plotly_chart(fig_auc, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Sampled Models (KNN & 5 SVM Kernels)")
        st.dataframe(SAMPLED_RESULTS, use_container_width=True, hide_index=True)
    with col_b:
        st.subheader("Regression Baselines (Threshold = 0.5)")
        st.dataframe(REGRESSION_RESULTS, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Dimensionality Reduction (PCA) & Unsupervised Clustering (K-Means)")
    st.markdown(
        """
        - **PCA:** **13 of 28** one-hot encoded features explain **95% of cumulative variance**.
        - **K-Means (k=2):** Chosen via Elbow plot and Silhouette score to mirror the binary Absence/Presence structure.
        - **Dataset Split:** 347,546 Absence (55.17%) vs. 282,454 Presence (44.83%), zero missing values, zero duplicates.
        """
    )
    for img_name, caption in [
        ("best_confusion_matrix.png", "Confusion Matrix — Logistic Regression (L1)"),
        ("best_roc_curve.png", "ROC Curve — Logistic Regression (L1)"),
        ("feature_importance.png", "Top 15 Features — Logistic Regression (L1)"),
        ("pca_variance.png", "PCA Cumulative Explained Variance"),
        ("kmeans_elbow_silhouette.png", "K-Means Elbow & Silhouette Scores"),
    ]:
        img_path = os.path.join("artifacts", img_name)
        if os.path.exists(img_path):
            st.image(img_path, caption=caption, use_container_width=True)