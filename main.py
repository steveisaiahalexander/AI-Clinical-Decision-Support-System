# ============================================================
#  Disease Symptom Classifier — Full Pipeline
#  Using Training.csv + Testing.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os

# === IMPORTANT: Add these sklearn imports properly ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder   # ← This was missing!
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

print("✅ All imports loaded successfully.\n")

# ── 1. Load Data ─────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Loading Training and Testing Data")
print("=" * 65)

train_df = pd.read_csv("Training.csv")
test_df = pd.read_csv("Testing.csv")

# Clean columns
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

symptom_cols = [col for col in train_df.columns if col != 'prognosis']

print(f"Training samples : {len(train_df)}")
print(f"Testing samples  : {len(test_df)}")
print(f"Symptoms         : {len(symptom_cols)}")
print(f"Diseases         : {train_df['prognosis'].nunique()}")

# ── 2. Preprocessing ─────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — Preprocessing")
print("=" * 65)

for col in symptom_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0).astype(int)
    test_df[col]  = pd.to_numeric(test_df[col],  errors='coerce').fillna(0).astype(int)

le = LabelEncoder()
train_df['target'] = le.fit_transform(train_df['prognosis'])
test_df['target']  = le.transform(test_df['prognosis'])

X_train = train_df[symptom_cols]
y_train = train_df['target']
X_test  = test_df[symptom_cols]
y_test  = test_df['target']

print("Data preprocessing completed.\n")

# ── 3. EDA Plots ─────────────────────────────────────────────
print("=" * 65)
print("STEP 3 — Generating EDA Plots")
print("=" * 65)

# Disease distribution
plt.figure(figsize=(12, 6))
train_df['prognosis'].value_counts().head(15).plot(kind='barh', color='teal')
plt.title("Top 15 Diseases in Training Set")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig("outputs/01_disease_distribution.png", dpi=150)
plt.close()

# Symptom frequency
symptom_freq = X_train.sum().sort_values(ascending=False).head(20)
plt.figure(figsize=(12, 8))
sns.barplot(x=symptom_freq.values, y=symptom_freq.index, palette="viridis")
plt.title("Top 20 Most Frequent Symptoms")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/02_top_symptoms.png", dpi=150)
plt.close()

print("EDA plots saved in outputs/ folder.\n")

# ── 4. Models Training & Evaluation ───────────────────────────
print("=" * 65)
print("STEP 4 — Training & Evaluating Models")
print("=" * 65)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Naive Bayes": GaussianNB()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"{name:15} → Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

# Best model (Random Forest)
best_model = models["Random Forest"]
best_pred = best_model.predict(X_test)
best_acc = results["Random Forest"]

print(f"\n🎯 Best Model: Random Forest with {best_acc:.4f} accuracy")

# ── 5. Hyperparameter Tuning ──────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — Hyperparameter Tuning (Random Forest)")
print("=" * 65)
print("This may take 1-2 minutes...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)

grid_search.fit(X_train, y_train)

tuned_model = grid_search.best_estimator_
tuned_pred = tuned_model.predict(X_test)
tuned_acc = accuracy_score(y_test, tuned_pred)

print(f"Best Parameters : {grid_search.best_params_}")
print(f"Tuned Test Accuracy : {tuned_acc:.4f}")

# Use tuned model as final
final_model = tuned_model if tuned_acc > best_acc else best_model

# ── 6. Confusion Matrix ───────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — Confusion Matrix")
print("=" * 65)

cm = confusion_matrix(y_test, final_model.predict(X_test))
plt.figure(figsize=(18, 16))
sns.heatmap(cm, annot=False, cmap="Blues", 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Final Model", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Disease")
plt.ylabel("Actual Disease")
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/03_confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix saved.")

# ── 7. Feature Importance ─────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7 — Top 20 Important Symptoms")
print("=" * 65)

importances = pd.Series(final_model.feature_importances_, index=symptom_cols)
top20 = importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x=top20.values, y=top20.index, palette="rocket")
plt.title("Top 20 Most Important Symptoms")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/04_feature_importance.png", dpi=150)
plt.close()

print(top20)

# ── 8. Save Final Model ───────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 8 — Saving Model & Encoder")
print("=" * 65)

joblib.dump(final_model, "outputs/final_disease_model.pkl")
joblib.dump(le, "outputs/label_encoder.pkl")
joblib.dump(symptom_cols, "outputs/symptom_columns.pkl")

print("✅ All files saved in outputs/ folder.")

# ── 9. Prediction Function ────────────────────────────────────
def predict_disease(symptoms_list, top_n=3):
    input_vec = pd.Series(0, index=symptom_cols)
    for sym in symptoms_list:
        clean = sym.strip().lower().replace(" ", "_").replace("-", "_")
        if clean in input_vec:
            input_vec[clean] = 1
    
    proba = final_model.predict_proba([input_vec.values])[0]
    top_idx = np.argsort(proba)[::-1][:top_n]
    return [(le.classes_[i], round(proba[i]*100, 2)) for i in top_idx]


# ── 10. Quick Demo ────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 9 — Quick Demo")
print("=" * 65)

sample_symptoms = ["itching", "skin_rash", "nodal_skin_eruptions"]
results = predict_disease(sample_symptoms, top_n=3)

print(f"Input Symptoms: {sample_symptoms}")
print("Top Predictions:")
for disease, confidence in results:
    print(f"  → {disease:<35} {confidence:6.2f}%")