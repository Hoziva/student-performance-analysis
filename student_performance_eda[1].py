# Student Performance Analysis and Prediction
# Course: Exploratory Data Analysis and Visualization
# Dataset: UCI Student Performance (student-mat.csv or student-por.csv)
# Run in Google Colab or a Python environment with internet access.
#
# Important:
# - This script downloads the public UCI dataset.
# - The target is G3 (final grade), converted into a binary classification target:
#   Pass = G3 >= 10, Fail = G3 < 10.
# - All performance values are generated when you run the script; do not report
#   results until you have executed it and checked the printed output.

import os
import urllib.request
import zipfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")
os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------
# 1. Download and load dataset
# ---------------------------------------------------------------------
zip_path = "student.zip"
data_dir = "student_data"
os.makedirs(data_dir, exist_ok=True)

url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
if not os.path.exists(zip_path):
    print("Downloading UCI Student Performance dataset...")
    urllib.request.urlretrieve(url, zip_path)

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(data_dir)

csv_path = os.path.join(data_dir, "student-mat.csv")
if not os.path.exists(csv_path):
    # Fallback for archives with a slightly different folder layout
    matches = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file == "student-mat.csv":
                matches.append(os.path.join(root, file))
    if not matches:
        raise FileNotFoundError("Could not find student-mat.csv inside downloaded archive.")
    csv_path = matches[0]

df = pd.read_csv(csv_path, sep=";")
print("\nDATASET SHAPE:", df.shape)
print("\nFIRST FIVE ROWS:\n", df.head())
print("\nCOLUMN NAMES:\n", df.columns.tolist())
print("\nDATA TYPES:\n", df.dtypes)
print("\nDESCRIPTIVE STATISTICS:\n", df.describe(include="all").T)
print("\nMISSING VALUES:\n", df.isna().sum())
print("\nEXACT DUPLICATE ROWS:", df.duplicated().sum())

# Save raw data summary for report evidence
df.describe(include="all").T.to_csv("outputs/descriptive_statistics.csv")
df.isna().sum().to_csv("outputs/missing_values.csv", header=["missing_count"])

# ---------------------------------------------------------------------
# 2. Data cleaning and target creation
# ---------------------------------------------------------------------
# Remove exact duplicate rows if any are present.
df = df.drop_duplicates().copy()

# The UCI data is mostly complete; categorical values are strings and
# numerical values are integers. Target is final grade G3.
if "G3" not in df.columns:
    raise KeyError("Expected final grade column G3 was not found.")

# Binary target: 1 = pass, 0 = fail. This threshold is defined for this study.
df["pass"] = (df["G3"] >= 10).astype(int)

print("\nPASS/FAIL COUNTS (0=Fail, 1=Pass):\n", df["pass"].value_counts().sort_index())
print("\nPASS/FAIL PROPORTIONS:\n", df["pass"].value_counts(normalize=True).sort_index())

# ---------------------------------------------------------------------
# 3. EDA visualizations (at least 5)
# ---------------------------------------------------------------------
# Visual 1: Final grade distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=20, kde=True)
plt.title("Figure 1. Distribution of Final Grades (G3)")
plt.xlabel("Final grade (0–20)")
plt.ylabel("Number of students")
plt.tight_layout()
plt.savefig("outputs/fig1_final_grade_distribution.png", dpi=200)
plt.show()

# Visual 2: Pass/fail class balance
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="pass")
plt.title("Figure 2. Student Pass/Fail Class Distribution")
plt.xlabel("Outcome (0 = Fail, 1 = Pass)")
plt.ylabel("Number of students")
plt.tight_layout()
plt.savefig("outputs/fig2_pass_fail_distribution.png", dpi=200)
plt.show()

# Visual 3: Study time vs final grade
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="studytime", y="G3")
plt.title("Figure 3. Final Grade by Weekly Study-Time Category")
plt.xlabel("Study time (1 = lowest, 4 = highest)")
plt.ylabel("Final grade (0–20)")
plt.tight_layout()
plt.savefig("outputs/fig3_studytime_vs_grade.png", dpi=200)
plt.show()

# Visual 4: Absences vs final grade
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="absences", y="G3", alpha=0.7)
plt.title("Figure 4. Absences and Final Grade")
plt.xlabel("Number of school absences")
plt.ylabel("Final grade (0–20)")
plt.tight_layout()
plt.savefig("outputs/fig4_absences_vs_grade.png", dpi=200)
plt.show()

# Visual 5: Correlation heatmap for numeric variables
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(13, 10))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Figure 5. Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig("outputs/fig5_correlation_heatmap.png", dpi=200)
plt.show()

# Visual 6: Grade by past failures
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="failures", y="G3")
plt.title("Figure 6. Final Grade by Number of Past Class Failures")
plt.xlabel("Number of past failures")
plt.ylabel("Final grade (0–20)")
plt.tight_layout()
plt.savefig("outputs/fig6_failures_vs_grade.png", dpi=200)
plt.show()

# Statistical relationships: correlations with G3, excluding the target itself
grade_correlations = numeric_df.corr(numeric_only=True)["G3"].drop(labels=["G3"], errors="ignore")
grade_correlations = grade_correlations.sort_values(key=lambda s: s.abs(), ascending=False)
print("\nNUMERIC FEATURE CORRELATIONS WITH G3:\n", grade_correlations)
grade_correlations.to_csv("outputs/correlation_with_final_grade.csv", header=["correlation_with_G3"])

# Outlier inspection using IQR for numeric columns (not automatically deleting them)
outlier_summary = []
for col in df.select_dtypes(include=np.number).columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((df[col] < lower) | (df[col] > upper)).sum())
    outlier_summary.append({"feature": col, "lower_IQR_bound": lower,
                            "upper_IQR_bound": upper, "outlier_count": count})
pd.DataFrame(outlier_summary).to_csv("outputs/iqr_outlier_summary.csv", index=False)
print("\nIQR OUTLIER SUMMARY SAVED TO outputs/iqr_outlier_summary.csv")
print("Outliers are flagged for inspection, not automatically removed, because extreme values may be valid.")

# ---------------------------------------------------------------------
# 4. Prepare model data
# ---------------------------------------------------------------------
# Remove G3 from inputs to avoid direct target leakage. G1 and G2 are earlier
# period grades and are retained; discuss that they may strongly predict G3.
X = df.drop(columns=["G3", "pass"])
y = df["pass"].astype(int)

# Stratified train/validation/test split: 70% / 15% / 15%.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

# Identify categorical/numerical columns based on training split.
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()

# Fit preprocessing only on training data to prevent information leakage.
X_train_num = X_train[numeric_cols].copy()
X_val_num = X_val[numeric_cols].copy()
X_test_num = X_test[numeric_cols].copy()

num_imputer = SimpleImputer(strategy="median")
X_train_num = num_imputer.fit_transform(X_train_num)
X_val_num = num_imputer.transform(X_val_num)
X_test_num = num_imputer.transform(X_test_num)

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_val_num = scaler.transform(X_val_num)
X_test_num = scaler.transform(X_test_num)

if categorical_cols:
    X_train_cat = X_train[categorical_cols].fillna("Unknown")
    X_val_cat = X_val[categorical_cols].fillna("Unknown")
    X_test_cat = X_test[categorical_cols].fillna("Unknown")

    # One-hot encode; align validation and test to training feature columns.
    X_train_cat = pd.get_dummies(X_train_cat, dtype=float)
    X_val_cat = pd.get_dummies(X_val_cat, dtype=float).reindex(columns=X_train_cat.columns, fill_value=0)
    X_test_cat = pd.get_dummies(X_test_cat, dtype=float).reindex(columns=X_train_cat.columns, fill_value=0)

    X_train_processed = np.hstack([X_train_num, X_train_cat.to_numpy()])
    X_val_processed = np.hstack([X_val_num, X_val_cat.to_numpy()])
    X_test_processed = np.hstack([X_test_num, X_test_cat.to_numpy()])
    feature_names = numeric_cols + list(X_train_cat.columns)
else:
    X_train_processed, X_val_processed, X_test_processed = X_train_num, X_val_num, X_test_num
    feature_names = numeric_cols

print("\nSPLIT SIZES:")
print("Training:", X_train_processed.shape, "Validation:", X_val_processed.shape, "Test:", X_test_processed.shape)
print("Input features after encoding:", len(feature_names))

# ---------------------------------------------------------------------
# 5. Deep learning model: Multilayer Perceptron (MLP)
# ---------------------------------------------------------------------
model = Sequential([
    Input(shape=(X_train_processed.shape[1],)),
    Dense(64, activation="relu"),
    Dropout(0.25),
    Dense(32, activation="relu"),
    Dropout(0.15),
    Dense(1, activation="sigmoid")
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

early_stop = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)
history = model.fit(
    X_train_processed, y_train,
    validation_data=(X_val_processed, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Visual 7: Training/validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("Figure 7. Training and Validation Loss by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy loss")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/fig7_loss_vs_epoch.png", dpi=200)
plt.show()

# Visual 8: Training/validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.title("Figure 8. Training and Validation Accuracy by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/fig8_accuracy_vs_epoch.png", dpi=200)
plt.show()

# ---------------------------------------------------------------------
# 6. Evaluate on held-out test data
# ---------------------------------------------------------------------
y_probability = model.predict(X_test_processed, verbose=0).ravel()
y_pred = (y_probability >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
try:
    roc_auc = roc_auc_score(y_test, y_probability)
except ValueError:
    roc_auc = float("nan")

print("\nHELD-OUT TEST RESULTS (GENERATED BY THIS RUN)")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("\nCLASSIFICATION REPORT:\n", classification_report(y_test, y_pred, zero_division=0))

metrics_df = pd.DataFrame([{
    "accuracy": accuracy, "precision": precision, "recall": recall,
    "f1_score": f1, "roc_auc": roc_auc, "epochs_run": len(history.history["loss"])
}])
metrics_df.to_csv("outputs/test_metrics.csv", index=False)

# Visual 9: Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Fail", "Predicted Pass"],
            yticklabels=["Actual Fail", "Actual Pass"])
plt.title("Figure 9. Test-Set Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.tight_layout()
plt.savefig("outputs/fig9_confusion_matrix.png", dpi=200)
plt.show()

# Visual 10: ROC curve
fpr, tpr, _ = roc_curve(y_test, y_probability)
curve_auc = auc(fpr, tpr)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"MLP ROC curve (AUC = {curve_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.title("Figure 10. ROC Curve on Test Data")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/fig10_roc_curve.png", dpi=200)
plt.show()

# ---------------------------------------------------------------------
# 7. Feature analysis (permutation importance on held-out test data)
# ---------------------------------------------------------------------
# Permutation importance measures the decrease in accuracy when a feature is
# shuffled. To keep this runnable for a small dataset, evaluate each encoded
# column individually using the trained network.
baseline_accuracy = accuracy_score(y_test, y_pred)
rng = np.random.default_rng(RANDOM_STATE)
importance_values = []

for j, feature_name in enumerate(feature_names):
    X_permuted = X_test_processed.copy()
    shuffled_col = X_permuted[:, j].copy()
    rng.shuffle(shuffled_col)
    X_permuted[:, j] = shuffled_col
    perm_probs = model.predict(X_permuted, verbose=0).ravel()
    perm_pred = (perm_probs >= 0.5).astype(int)
    perm_accuracy = accuracy_score(y_test, perm_pred)
    importance_values.append(baseline_accuracy - perm_accuracy)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "accuracy_decrease_after_permutation": importance_values
}).sort_values("accuracy_decrease_after_permutation", ascending=False)
importance_df.to_csv("outputs/permutation_feature_importance.csv", index=False)

top_n = min(15, len(importance_df))
plt.figure(figsize=(10, 6))
sns.barplot(
    data=importance_df.head(top_n),
    x="accuracy_decrease_after_permutation",
    y="feature"
)
plt.title("Figure 11. Top Permutation Feature Importance (Test Accuracy Decrease)")
plt.xlabel("Decrease in accuracy after shuffling feature")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("outputs/fig11_feature_importance.png", dpi=200)
plt.show()

print("\nTOP FEATURES BY PERMUTATION IMPORTANCE:\n", importance_df.head(15))
print("\nOUTPUT FILES SAVED IN ./outputs/")
print("Use these real outputs to complete the Results and Interpretation sections of your report.")
