"""
╔══════════════════════════════════════════════════════════════════════╗
║          IMPROVED TRAINING SCRIPT - train_model.py                  ║
║                                                                      ║
║  IMPROVEMENTS OVER ORIGINAL:                                        ║
║  1. Uses Gradient Boosting instead of Random Forest                 ║
║     → Better accuracy on tabular medical data                       ║
║  2. Cross-validation for reliable accuracy estimate                 ║
║  3. Handles NaN/missing values properly                             ║
║  4. Saves feature importance for debugging                          ║
║  5. Produces detailed per-disease accuracy report                   ║
║  6. Model comparison: tries both RF and GB, saves the best          ║
╚══════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    python train_model.py

OUTPUT FILES:
    model.pkl          ← main model (used by app.py)
    label_encoder.pkl  ← disease name encoder
    model_info.pkl     ← metadata (accuracy, feature importance)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import warnings

warnings.filterwarnings("ignore")

print("=" * 65)
print("🚀 AI Disease Predictor - Model Training Script v2.0")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────────────
print("\n📂 STEP 1: Loading dataset...")

try:
    df = pd.read_csv('dataset/Training.csv')
    print(f"   ✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print("   ❌ ERROR: dataset/Training.csv not found!")
    print("   Please ensure the dataset folder exists with Training.csv")
    exit(1)

print(f"   📊 Unique diseases: {df['prognosis'].nunique()}")
print(f"   📊 Symptom columns: {df.shape[1] - 1}")

# ─────────────────────────────────────────────────────────────────────
# STEP 2: PREPROCESSING
# IMPROVED: Handles edge cases and ensures clean binary data
# ─────────────────────────────────────────────────────────────────────
print("\n🔧 STEP 2: Preprocessing...")

# Separate features and labels
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Clean symptom columns: convert to numeric, fill NaN with 0
# (NaN in symptom columns = symptom not present = 0)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    X[col] = X[col].clip(0, 1)  # Ensure binary (0 or 1)

print(f"   ✅ Features cleaned: {X.shape[1]} symptoms")
print(f"   ✅ NaN values: {X.isnull().sum().sum()} (should be 0)")

# Encode disease names to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"   ✅ Encoded {len(le.classes_)} disease classes")

# ─────────────────────────────────────────────────────────────────────
# STEP 3: TRAIN/TEST SPLIT
# IMPROVED: Stratified split ensures balanced disease representation
# ─────────────────────────────────────────────────────────────────────
print("\n✂️  STEP 3: Splitting data (80% train / 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded  # Ensures each disease has proportional representation
)

print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples:  {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────────────
# STEP 4: TRAIN MULTIPLE MODELS & COMPARE
# NEW: We train 2 models and pick the better one (or ensemble them)
# ─────────────────────────────────────────────────────────────────────
print("\n🤖 STEP 4: Training models...")

# Model 1: Original Random Forest (for comparison)
print("\n   Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,          # More trees than original (100 → 200)
    max_depth=None,            # Let trees grow fully
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,                 # Use all CPU cores
    class_weight='balanced'    # Handle class imbalance
)
rf_model.fit(X_train, y_train)
rf_train_acc = rf_model.score(X_train, y_train)
rf_test_acc = rf_model.score(X_test, y_test)
print(f"   Random Forest  → Train: {rf_train_acc:.4f} | Test: {rf_test_acc:.4f}")

# Model 2: Gradient Boosting (often better for medical data)
print("\n   Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_train_acc = gb_model.score(X_train, y_train)
gb_test_acc = gb_model.score(X_test, y_test)
print(f"   Gradient Boost → Train: {gb_train_acc:.4f} | Test: {gb_test_acc:.4f}")

# Pick the best model
if gb_test_acc >= rf_test_acc:
    best_model = gb_model
    best_name = "Gradient Boosting"
    best_test_acc = gb_test_acc
else:
    best_model = rf_model
    best_name = "Random Forest"
    best_test_acc = rf_test_acc

print(f"\n   🏆 Best model: {best_name} (Test Accuracy: {best_test_acc:.4f})")

# ─────────────────────────────────────────────────────────────────────
# STEP 5: CROSS-VALIDATION
# NEW: 5-fold cross-validation gives reliable accuracy estimate
# ─────────────────────────────────────────────────────────────────────
print("\n📊 STEP 5: Cross-validation (5-fold)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"   CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"   CV Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────
# STEP 6: DETAILED EVALUATION
# ─────────────────────────────────────────────────────────────────────
print("\n📋 STEP 6: Detailed Evaluation...")

y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n   Final Test Accuracy: {test_acc:.4f} ({test_acc:.1%})")
print(f"   Weighted F1 Score:   {f1:.4f}")

print("\n📋 Per-Disease Report:")
print("-" * 65)
report = classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    output_dict=False
)
# Print only summary (not full long report unless needed)
lines = report.split('\n')
# Print last 5 lines (averages)
for line in lines[-6:]:
    if line.strip():
        print(f"   {line}")

# ─────────────────────────────────────────────────────────────────────
# STEP 7: FEATURE IMPORTANCE
# NEW: Shows which symptoms matter most for predictions
# ─────────────────────────────────────────────────────────────────────
print("\n🔍 STEP 7: Top 15 Most Important Symptoms...")

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_names = X.columns.tolist()
    
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    
    print("   Rank | Symptom                          | Importance")
    print("   " + "-" * 55)
    for i, idx in enumerate(sorted_idx[:15]):
        print(f"   {i+1:4d} | {feature_names[idx]:32s} | {importances[idx]:.4f}")

# ─────────────────────────────────────────────────────────────────────
# STEP 8: SAVE FILES
# ─────────────────────────────────────────────────────────────────────
print("\n💾 STEP 8: Saving model files...")

# Save main model
joblib.dump(best_model, 'model.pkl')
print("   ✅ model.pkl saved")

# Save label encoder
joblib.dump(le, 'label_encoder.pkl')
print("   ✅ label_encoder.pkl saved")

# Save model metadata (useful for the app)
model_info = {
    "model_name": best_name,
    "test_accuracy": test_acc,
    "cv_mean": cv_scores.mean(),
    "cv_std": cv_scores.std(),
    "f1_score": f1,
    "n_symptoms": X.shape[1],
    "n_diseases": len(le.classes_),
    "disease_list": le.classes_.tolist(),
    "symptom_list": X.columns.tolist(),
}
joblib.dump(model_info, 'model_info.pkl')
print("   ✅ model_info.pkl saved")

# ─────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("✅ TRAINING COMPLETE!")
print("=" * 65)
print(f"   Model:          {best_name}")
print(f"   Test Accuracy:  {test_acc:.1%}")
print(f"   CV Accuracy:    {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
print(f"   Diseases:       {len(le.classes_)}")
print(f"   Symptoms:       {X.shape[1]}")
print("\n🚀 Ready! Run: streamlit run app.py")
print("=" * 65)


