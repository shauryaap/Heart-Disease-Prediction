import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sklearn

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    RocCurveDisplay,
    r2_score,
    mean_squared_error,
    silhouette_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Setup, Configuration and Data Loading
# -----------------------------------------------------------------------------
QUICK_RUN = False  # Set True for a fast test run on a small subset
RANDOM_STATE = 42
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

if QUICK_RUN:
    MAX_ROWS = 60000
    N_TREES_RF, N_TREES_BAG, N_TREES_ADA = 40, 10, 30
    SAMPLED_TRAIN_N, SAMPLED_TEST_N = 8000, 4000
    SVM_KERNEL_TRAIN_N, SVM_KERNEL_TEST_N = 3000, 1500
    PCA_N, CLUSTER_N, CV_N = 6000, 3000, 15000
else:
    MAX_ROWS = None
    N_TREES_RF, N_TREES_BAG, N_TREES_ADA = 150, 25, 50
    SAMPLED_TRAIN_N, SAMPLED_TEST_N = 30000, 15000
    SVM_KERNEL_TRAIN_N, SVM_KERNEL_TEST_N = 8000, 4000
    PCA_N, CLUSTER_N, CV_N = 15000, 8000, 40000


def save_fig(name):
    """Save the current figure for the Streamlit app."""
    plt.savefig(os.path.join(ARTIFACT_DIR, name), dpi=130, bbox_inches="tight")


def load_dataset(data_file="train.csv"):
    df = pd.read_csv(data_file)
    if MAX_ROWS and len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    print("scikit-learn version:", sklearn.__version__)
    print("Dataset shape:", df.shape)
    return df


if __name__ == "__main__":
    df = load_dataset("train.csv")

    # -------------------------------------------------------------------------
    # 2. Inspect and Clean the Data
    # -------------------------------------------------------------------------
    TARGET = "Heart Disease"
    df = df.drop(columns=["id"], errors="ignore")
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = (
        df[TARGET].astype(str).str.strip().str.lower().map({"absence": 0, "presence": 1})
    )
    if df[TARGET].isna().any():
        raise ValueError("Unexpected target labels in the data.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    print("Cleaned shape:", df.shape, "| Predictors:", X.shape[1])

    # -------------------------------------------------------------------------
    # 3. Exploratory Data Analysis
    # -------------------------------------------------------------------------
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Absence (0)", "Presence (1)"], counts.values)
    plt.title("Heart Disease Target Distribution")
    plt.ylabel("Records")
    plt.tight_layout()
    save_fig("eda_target.png")
    plt.close()

    X.hist(figsize=(16, 12), bins=30, edgecolor="black")
    plt.suptitle("Distribution of Predictors", fontsize=16)
    plt.tight_layout()
    save_fig("eda_distributions.png")
    plt.close()

    corr_data = X.copy()
    corr_data[TARGET] = y
    corr = corr_data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    save_fig("eda_correlation.png")
    plt.close()

    # -------------------------------------------------------------------------
    # 4. Stratified 80:20 Train-Test Split
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # -------------------------------------------------------------------------
    # 5. Preprocessing
    # -------------------------------------------------------------------------
    categorical_cols = [
        c
        for c in [
            "Sex",
            "Chest pain type",
            "FBS over 120",
            "EKG results",
            "Exercise angina",
            "Slope of ST",
            "Number of vessels fluro",
            "Thallium",
        ]
        if c in X.columns
    ]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    def make_preprocessor():
        num = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        cat = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            [("num", num, numeric_cols), ("cat", cat, categorical_cols)],
            remainder="drop",
        )

    def pipe(estimator):
        return Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])

    def make_hgb():
        pre = ColumnTransformer(
            [
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=np.nan
                    ),
                    categorical_cols,
                )
            ],
            remainder="passthrough",
        )
        mask = [True] * len(categorical_cols) + [False] * len(numeric_cols)
        return Pipeline(
            [
                ("preprocess", pre),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=150,
                        learning_rate=0.08,
                        max_leaf_nodes=31,
                        l2_regularization=1.0,
                        categorical_features=mask,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    # -------------------------------------------------------------------------
    # 6 & 7. Classification, Ensemble, KNN & Linear SVM
    # -------------------------------------------------------------------------
    models = {
        "Logistic Regression (L2)": pipe(
            LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000)
        ),
        "Logistic Regression (L1)": pipe(
            LogisticRegression(penalty="l1", solver="saga", max_iter=1000)
        ),
        "Decision Tree": pipe(
            DecisionTreeClassifier(
                max_depth=15,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "Random Forest": pipe(
            RandomForestClassifier(
                n_estimators=N_TREES_RF,
                max_depth=20,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample",
            )
        ),
        "Bagging": pipe(
            BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=12, min_samples_leaf=10, random_state=RANDOM_STATE
                ),
                n_estimators=N_TREES_BAG,
                max_samples=0.8,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "AdaBoost (Boosting)": pipe(
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
                n_estimators=N_TREES_ADA,
                learning_rate=0.5,
                random_state=RANDOM_STATE,
            )
        ),
        "HistGradientBoosting (Boosting)": make_hgb(),
    }

    results = []
    fitted_models = {}

    def evaluate_model(name, model, x_eval, y_eval, label, group):
        pred = model.predict(x_eval)
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(x_eval)[:, 1]
        elif hasattr(model, "decision_function"):
            score = model.decision_function(x_eval)
        else:
            score = None
        results.append(
            {
                "Model": name,
                "Group": group,
                "Evaluation": label,
                "Accuracy": accuracy_score(y_eval, pred),
                "Precision": precision_score(y_eval, pred, zero_division=0),
                "Recall": recall_score(y_eval, pred, zero_division=0),
                "F1": f1_score(y_eval, pred, zero_division=0),
                "ROC_AUC": roc_auc_score(y_eval, score) if score is not None else np.nan,
            }
        )

    for name, model in models.items():
        print("Training:", name)
        model.fit(X_train, y_train)
        fitted_models[name] = model
        evaluate_model(name, model, X_test, y_test, "Full 20% test split", "Full")

    # -------------------------------------------------------------------------
    # 9. Cross-Validation Model Selection & Best Model Export
    # -------------------------------------------------------------------------
    rng = np.random.RandomState(RANDOM_STATE)
    cv_idx = rng.choice(len(X_train), min(CV_N, len(X_train)), replace=False)
    X_cv, y_cv = X_train.iloc[cv_idx], y_train.iloc[cv_idx]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    full_df = pd.DataFrame([r for r in results if r["Group"] == "Full"])
    full_df.to_csv(os.path.join(ARTIFACT_DIR, "model_comparison.csv"), index=False)

    cv_rows = []
    for name in full_df["Model"]:
        est = clone(fitted_models[name])
        sc = cross_val_score(est, X_cv, y_cv, cv=skf, scoring="roc_auc")
        cv_rows.append(
            {"Model": name, "CV_ROC_AUC_mean": sc.mean(), "CV_ROC_AUC_std": sc.std()}
        )

    cv_df = pd.DataFrame(cv_rows).sort_values("CV_ROC_AUC_mean", ascending=False)
    cv_df.to_csv(os.path.join(ARTIFACT_DIR, "cv_results.csv"), index=False)

    best_name = cv_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "best_heart_disease_model.joblib"))
    print("Saved best model:", best_name)