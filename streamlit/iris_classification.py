"""
IRIS FLOWER CLASSIFICATION - COMPLETE ML PROJECT
A comprehensive machine learning solution for classifying iris species
Run this file to execute the complete analysis pipeline
"""

# ============================================================================
# STEP 1: IMPORT LIBRARIES AND LOAD DATASET
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("IRIS FLOWER CLASSIFICATION PROJECT")
print("="*80)
print("\n>>> STEP 1: LOADING DATASET\n")

# Load dataset from Excel file
df = pd.read_csv("streamlit/iris.csv")

# Check the first few rows
print("First 5 rows of dataset:")
print(df.head())

# Basic info
print("\nDataset shape:", df.shape)

# Encode species names to numbers (0,1,2)
encoder = LabelEncoder()
df['species_encoded'] = encoder.fit_transform(df['Species'])

# Separate features (X) and target (y)
X = df.iloc[:, 0:4].values
y = df['species_encoded'].values

print("\nDataset loaded successfully!")
print("Features:", list(df.columns[:4]))
print("Target classes:", list(encoder.classes_))
# ============================================================================
# STEP 2: DATA EXPLORATION AND VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print(">>> STEP 2: DATA EXPLORATION\n")

print("Checking for missing values:")
print(df.isnull().sum())
print("\n* No missing values found!")

print("\n" + "-"*80)
print("DESCRIPTIVE STATISTICS:")
print("-"*80)
print(df.describe())

print("\n" + "-"*80)
print("CLASS DISTRIBUTION:")
print("-"*80)
print(df['Species'].value_counts())

correlation_matrix = df.iloc[:, 0:4].corr()
print("\n" + "-"*80)
print("CORRELATION MATRIX:")
print("-"*80)
print(correlation_matrix)

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print(">>> STEP 3: DATA PREPROCESSING\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n* Features standardized (mean=0, std=1)")

# ============================================================================
# STEP 4: MODEL TRAINING
# ============================================================================

print("\n" + "="*80)
print(">>> STEP 4: MODEL TRAINING\n")

models = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42)
}

trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

print("\n* All models trained!")

# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================

print("\n" + "="*80)
print(">>> STEP 5: MODEL EVALUATION\n")

results = []

for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

print("\n" + "="*80)
print("MODEL COMPARISON:")
print("="*80)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]

print(f"\n BEST MODEL: {best_model_name}")

# Confusion Matrix
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('visualizations/iris_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n* Saved: visualizations/iris_confusion_matrix.png")

# =====================================================================
# STEP 6: SAVE THE BEST MODEL, SCALER, AND ENCODER FOR DEPLOYMENT
# =====================================================================

import pickle

# ✅ If you already know your best model is SVM:
best_model_name = "Support Vector Machine"
best_model = models[best_model_name]  # pick SVM model from your models dictionary

# ✅ If you have fine-tuned models, and want to use the tuned one instead:
# best_model = tuned_models['Support Vector Machine']

# ✅ Prepare everything needed for deployment
deployment_objects = {
    'model': best_model,
    'scaler': scaler,
    'encoder': encoder
}

# ✅ Save all in one file
with open("iris_svm_model.pkl", "wb") as f:
    pickle.dump(deployment_objects, f)

print(f"\n* Successfully saved the BEST model ({best_model_name}), scaler, and encoder to iris_svm_model.pkl.")
