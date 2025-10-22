"""
COMPREHENSIVE MODEL COMPARISON
Compare multiple models and visualize decision boundaries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

# Load data
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)

# Define models
models = {
    'KNN (K=3)': KNeighborsClassifier(n_neighbors=3),
    'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN (K=7)': KNeighborsClassifier(n_neighbors=7),
    'Decision Tree (depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Decision Tree (depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest (50)': RandomForestClassifier(n_estimators=50, random_state=42),
    'Random Forest (100)': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (Linear)': SVC(kernel='linear', random_state=42),
    'SVM (RBF, C=1)': SVC(kernel='rbf', C=1, random_state=42),
    'SVM (RBF, C=10)': SVC(kernel='rbf', C=10, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
}

print("\n>>> Evaluating all model configurations...\n")

results = []

for name, model in models.items():
    # Training time
    start = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = (time.time() - start) * 1000
    
    # Prediction time
    start = time.time()
    y_pred = model.predict(X_test_scaled)
    pred_time = (time.time() - start) * 1000
    
    # Metrics
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    results.append({
        'Model': name,
        'Train Acc': train_acc,
        'Test Acc': test_acc,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Train Time (ms)': train_time,
        'Pred Time (ms)': pred_time,
        'Overfit': train_acc - test_acc
    })

results_df = pd.DataFrame(results)

print("="*80)
print("DETAILED MODEL COMPARISON")
print("="*80)
print(results_df.to_string(index=False))

# Sort by test accuracy
results_df_sorted = results_df.sort_values('Test Acc', ascending=False)

print("\n" + "="*80)
print("TOP 5 MODELS BY TEST ACCURACY")
print("="*80)
print(results_df_sorted.head().to_string(index=False))

# Visualization: Model comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comprehensive Model Analysis', fontsize=18, fontweight='bold')

# Plot 1: Test Accuracy
ax1 = axes[0, 0]
sorted_results = results_df.sort_values('Test Acc')
colors = ['red' if x < 0.95 else 'orange' if x < 0.97 else 'green' 
          for x in sorted_results['Test Acc']]
ax1.barh(range(len(sorted_results)), sorted_results['Test Acc'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(sorted_results)))
ax1.set_yticklabels(sorted_results['Model'], fontsize=8)
ax1.set_xlabel('Test Accuracy', fontweight='bold')
ax1.set_title('Model Test Accuracy', fontweight='bold')
ax1.axvline(x=0.95, color='red', linestyle='--', alpha=0.5)
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Cross-Validation with Error Bars
ax2 = axes[0, 1]
sorted_cv = results_df.sort_values('CV Mean')
ax2.barh(range(len(sorted_cv)), sorted_cv['CV Mean'], 
         xerr=sorted_cv['CV Std'], color='steelblue', alpha=0.7, capsize=3)
ax2.set_yticks(range(len(sorted_cv)))
ax2.set_yticklabels(sorted_cv['Model'], fontsize=8)
ax2.set_xlabel('CV Mean Accuracy', fontweight='bold')
ax2.set_title('Cross-Validation Performance', fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Overfitting Analysis
ax3 = axes[1, 0]
ax3.scatter(results_df['Train Acc'], results_df['Test Acc'], 
           s=100, alpha=0.6, c=results_df['Overfit'], cmap='RdYlGn_r')
ax3.plot([0.9, 1.0], [0.9, 1.0], 'k--', alpha=0.5)
ax3.set_xlabel('Training Accuracy', fontweight='bold')
ax3.set_ylabel('Test Accuracy', fontweight='bold')
ax3.set_title('Overfitting Analysis', fontweight='bold')
ax3.grid(alpha=0.3)

# Plot 4: Speed vs Accuracy
ax4 = axes[1, 1]
scatter = ax4.scatter(results_df['Pred Time (ms)'], results_df['Test Acc'],
                     s=100, alpha=0.6, c=results_df['Test Acc'], cmap='viridis')
ax4.set_xlabel('Prediction Time (ms)', fontweight='bold')
ax4.set_ylabel('Test Accuracy', fontweight='bold')
ax4.set_title('Speed vs Accuracy Trade-off', fontweight='bold')
ax4.set_xscale('log')
ax4.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Test Accuracy')

plt.tight_layout()
plt.savefig('visualizations/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n* Saved: visualizations/comprehensive_model_comparison.png")

# Decision boundaries visualization (2D)
print("\n>>> Generating decision boundary visualizations...")

# Use only petal features for 2D visualization
X_2d = X[:, [2, 3]]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)

scaler_2d = StandardScaler()
X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
X_test_2d_scaled = scaler_2d.transform(X_test_2d)

# Select representative models
viz_models = {
    'KNN (K=3)': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', C=10, random_state=42),
}

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Decision Boundaries (Petal Length vs Petal Width)', 
             fontsize=16, fontweight='bold')

# Create mesh
h = 0.02
x_min, x_max = X_train_2d_scaled[:, 0].min() - 1, X_train_2d_scaled[:, 0].max() + 1
y_min, y_max = X_train_2d_scaled[:, 1].min() - 1, X_train_2d_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

for idx, (name, model) in enumerate(viz_models.items()):
    ax = axes[idx // 2, idx % 2]
    
    model.fit(X_train_2d_scaled, y_train_2d)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
    ax.scatter(X_train_2d_scaled[:, 0], X_train_2d_scaled[:, 1],
              c=y_train_2d, cmap='Set1', edgecolor='black', s=50, alpha=0.8)
    
    test_acc = model.score(X_test_2d_scaled, y_test_2d)
    ax.set_xlabel('Petal Length (scaled)', fontweight='bold')
    ax.set_ylabel('Petal Width (scaled)', fontweight='bold')
    ax.set_title(f'{name} (Accuracy: {test_acc:.2%})', fontweight='bold')

legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=plt.cm.Set1(i/2), markersize=10,
                             label=iris.target_names[i]) 
                  for i in range(3)]
axes[0, 0].legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig('visualizations/decision_boundaries.png', dpi=300, bbox_inches='tight')
print("* Saved: visualizations/decision_boundaries.png")

# Summary
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

best_idx = results_df['Test Acc'].idxmax()
best_model_info = results_df.loc[best_idx]
print(f"\nBest Model: {best_model_info['Model']}")
print(f"  Test Accuracy: {best_model_info['Test Acc']:.4f}")
print(f"  CV Score: {best_model_info['CV Mean']:.4f} (±{best_model_info['CV Std']:.4f})")

fast_idx = results_df['Pred Time (ms)'].idxmin()
fast_model_info = results_df.loc[fast_idx]
print(f"\nFastest Model: {fast_model_info['Model']}")
print(f"  Prediction Time: {fast_model_info['Pred Time (ms)']:.4f} ms")
print(f"  Test Accuracy: {fast_model_info['Test Acc']:.4f}")

print("\n" + "="*80)
print("KEY RECOMMENDATIONS")
print("="*80)
print("""
FOR MAXIMUM ACCURACY: Use SVM (RBF, C=10) or Random Forest (100 trees)
FOR SPEED:            Use Logistic Regression or KNN
FOR INTERPRETABILITY: Use Decision Tree
FOR ROBUSTNESS:       Use Random Forest or Ensemble methods
""")

print("\n* Comprehensive model comparison completed!")
print("="*80)