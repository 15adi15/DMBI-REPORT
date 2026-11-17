# ==============================================================================
# DMBI CA2 - MINI PROJECT IMPLEMENTATION (HEALTHCARE CLINICAL DECISION SUPPORT)
# Topic: Predictive Clinical Screening & Metabolic Risk Clustering for Diabetes
# Team: Aditya Bhore (05), Divya Dhameja (10), Ishan Katariya (19), Raj Chavan (07)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, silhouette_score
)

# Set visual style
plt.rcParams['figure.dpi'] = 150
sns.set_theme(style="whitegrid", palette="deep")

# ------------------------------------------------------------------------------
# 1. DATA LOADING & CLINICAL UNDERSTANDING
# ------------------------------------------------------------------------------
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

print("=== 1. RAW DATASET SHAPE & OVERVIEW ===")
print(f"Total Patient Records: {df.shape[0]} | Total Clinical Attributes: {df.shape[1]}")
print(df.head())

# ------------------------------------------------------------------------------
# 2. DATA PREPROCESSING PIPELINE (MODULE II)
# ------------------------------------------------------------------------------
print("\n=== 2. DATA PREPROCESSING PIPELINE (MODULE II) ===")

# A. Identify biologically invalid 0 values in physiological columns
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
zero_counts = (df[zero_cols] == 0).sum()
print("Biologically Invalid Zero Values Detected per Column:")
print(zero_counts.to_dict(), f"| Total Invalid Zeros = {zero_counts.sum()}")

# Replace 0 with NaN for physiological lab columns
df_clean = df.copy()
df_clean[zero_cols] = df_clean[zero_cols].replace(0, np.nan)

# B. Separate Predictors (X) and Target (y)
X = df_clean.drop(columns=['Outcome'])
y = df_clean['Outcome']

# C. Stratified Train-Test Split (80% Train = 614 rows, 20% Test = 154 rows)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# D. Robust Median Imputation (prevents skew from extreme Insulin outliers)
imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=X.columns)

# E. Z-Score Standardization
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test_imp), columns=X.columns)

print(f"Processed Feature Matrix Shape: {X.shape}")
print(f"Training Set: {X_train.shape[0]} rows | Testing Set: {X_test.shape[0]} rows")

# ------------------------------------------------------------------------------
# 3. SUPERVISED CLASSIFICATION (MODULE III: DECISION TREE & RANDOM FOREST)
# ------------------------------------------------------------------------------
# Algorithm 1: Pruned Decision Tree (CART)
dt_model = DecisionTreeClassifier(
    max_depth=4, criterion='gini', class_weight='balanced', random_state=42
)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

# Algorithm 2: Random Forest Classifier (200-Tree Ensemble)
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=6, min_samples_leaf=8,
    class_weight='balanced', random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------------------------
# 4. MODEL EVALUATION & COMPARISON
# ------------------------------------------------------------------------------
def evaluate_model(name, y_true, y_pred, y_prob):
    return {
        "Algorithm": name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall (Sensitivity)": round(recall_score(y_true, y_pred), 4),
        "F1-Score": round(f1_score(y_true, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 4)
    }

results_df = pd.DataFrame([
    evaluate_model("Decision Tree (CART)", y_test, y_pred_dt, y_prob_dt),
    evaluate_model("Random Forest (Ensemble)", y_test, y_pred_rf, y_prob_rf)
])

print("\n=== 3. MODEL EVALUATION TABLE (HOLDOUT TEST SET = 154 PATIENTS) ===")
print(results_df.to_string(index=False))

# ------------------------------------------------------------------------------
# 5. UNSUPERVISED CLUSTERING (MODULE IV: K-MEANS METABOLIC PHENOTYPES)
# ------------------------------------------------------------------------------
print("\n=== 4. K-MEANS METABOLIC PHENOTYPE SEGMENTATION (268 DIABETIC PATIENTS) ===")
X_all_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
diabetics_df = X_all_imp[y == 1][['Glucose', 'BMI', 'Insulin', 'Age']].copy()

cluster_scaler = StandardScaler()
diabetics_scaled = cluster_scaler.fit_transform(diabetics_df[['Glucose', 'BMI', 'Age']])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
diabetics_df['Cluster'] = kmeans.fit_predict(diabetics_scaled)
sil_score = silhouette_score(diabetics_scaled, diabetics_df['Cluster'])
print(f"K-Means (K=3) Silhouette Score on Confirmed Diabetic Patients: {sil_score:.4f}")

cluster_summary = diabetics_df.groupby('Cluster')[['Glucose', 'BMI', 'Insulin', 'Age']].mean().round(2)
cluster_summary['Patient_Count'] = diabetics_df['Cluster'].value_counts()
print("\nCluster Metabolic Profiles (Mean Values):")
print(cluster_summary)

# ------------------------------------------------------------------------------
# 6. GENERATE 5 CLINICAL VISUALIZATIONS FOR CA2 REPORT
# ------------------------------------------------------------------------------
# Chart 1: Diabetes Distribution by Clinical Glucose Risk Band
df_viz = X_all_imp.copy()
df_viz['Outcome'] = y.map({0: 'Non-Diabetic (0)', 1: 'Diabetic (1)'})
df_viz['Glucose_Band'] = pd.cut(
    df_viz['Glucose'],
    bins=[0, 99, 125, 300],
    labels=['Normal (<100 mg/dL)', 'Prediabetic (100-125)', 'Hyperglycemic (>=126)']
)

plt.figure(figsize=(7.5, 4.5))
ax = sns.countplot(data=df_viz, x='Glucose_Band', hue='Outcome', palette=['#0f766e', '#ea580c'])
plt.title("Visual 1: Patient Diabetes Distribution by Clinical Glucose Risk Category", fontweight='bold')
plt.xlabel("2-Hour OGTT Plasma Glucose Category")
plt.ylabel("Number of Patients")
for container in ax.containers:
    ax.bar_label(container)
plt.tight_layout()
plt.savefig("CA2_Healthcare_Chart1_Glucose_Bands.png")
plt.show()

# Chart 2: Confusion Matrices (Decision Tree vs Random Forest)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Decision Tree (CART) Confusion Matrix", fontweight='bold')
axes[0].set_xlabel("Predicted Diagnosis")
axes[0].set_ylabel("Actual Diagnosis")

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='YlGnBu', ax=axes[1])
axes[1].set_title("Random Forest Confusion Matrix", fontweight='bold')
axes[1].set_xlabel("Predicted Diagnosis")
axes[1].set_ylabel("Actual Diagnosis")
plt.tight_layout()
plt.savefig("CA2_Healthcare_Chart2_Confusion_Matrices.png")
plt.show()

# Chart 3: ROC-AUC Curve Comparison
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(7, 5))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.3f})", linewidth=2.5, color='#0f766e')
plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {roc_auc_score(y_test, y_prob_dt):.3f})", linewidth=2, linestyle='--', color='#d97706')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random Guess (AUC = 0.500)")
plt.title("Visual 3: ROC Curve Comparison (Random Forest vs. Decision Tree)", fontweight='bold')
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Recall / Sensitivity)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("CA2_Healthcare_Chart3_ROC_Curves.png")
plt.show()

# Chart 4: Random Forest Clinical Feature Importance
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
plt.figure(figsize=(8, 4.8))
feat_imp.plot(kind='barh', color='#0f766e')
plt.title("Visual 4: Clinical Drivers of Diabetes (Random Forest Gini Importance)", fontweight='bold')
plt.xlabel("Relative Gini Importance Score")
plt.tight_layout()
plt.savefig("CA2_Healthcare_Chart4_Feature_Importance.png")
plt.show()

# Chart 5: K-Means Diabetic Phenotype Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=diabetics_df, x='Glucose', y='BMI',
    hue='Cluster', palette=['#db2777', '#7c3aed', '#ea580c'], s=65, alpha=0.85
)
plt.title("Visual 5: K-Means Clustering of Confirmed Diabetic Patients (K = 3 Phenotypes)", fontweight='bold')
plt.xlabel("Plasma Glucose Concentration (mg/dL)")
plt.ylabel("Body Mass Index (BMI kg/m^2)")
plt.legend(title="Phenotype Cluster")
plt.tight_layout()
plt.savefig("CA2_Healthcare_Chart5_KMeans_Phenotypes.png")
plt.show()