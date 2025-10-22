"""
FEATURE IMPORTANCE ANALYSIS
Understand which features contribute most to predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

print("="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Method 1: Random Forest Feature Importance
print("\n>>> METHOD 1: Random Forest Feature Importance")
print("-"*80)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
feature_names = iris.feature_names

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nFeature Importance Scores:")
print(importance_df.to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold', pad=20)
plt.gca().invert_yaxis()

for i, (bar, value) in enumerate(zip(bars, importance_df['Importance'])):
    plt.text(value + 0.01, i, f'{value:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/feature_importance_rf.png', dpi=300, bbox_inches='tight')
print("\n* Saved: visualizations/feature_importance_rf.png")

# Method 2: Permutation Importance
print("\n>>> METHOD 2: Permutation Importance")
print("-"*80)

perm_importance = permutation_importance(
    rf_model, X_test, y_test, n_repeats=30, random_state=42
)

perm_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

print("\nPermutation Importance (with std deviation):")
for _, row in perm_importance_df.iterrows():
    print(f"  {row['Feature']:25} {row['Importance']:.4f} (±{row['Std']:.4f})")

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(perm_importance_df['Feature'], perm_importance_df['Importance'], 
         xerr=perm_importance_df['Std'], color='coral', capsize=5)
plt.xlabel('Permutation Importance', fontsize=12, fontweight='bold')
plt.title('Feature Importance with Uncertainty', fontsize=14, fontweight='bold', pad=20)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('visualizations/feature_importance_permutation.png', dpi=300, bbox_inches='tight')
print("\n* Saved: visualizations/feature_importance_permutation.png")

# Method 3: Correlation with Target
print("\n>>> METHOD 3: Correlation with Target Variable")
print("-"*80)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

correlations = df.corr()['target'].drop('target').sort_values(ascending=False)

print("\nCorrelation with Species:")
for feature, corr in correlations.items():
    print(f"  {feature:25} {corr:+.4f}")

plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(correlations)), correlations.values, color='mediumseagreen')
plt.xticks(range(len(correlations)), correlations.index, rotation=45, ha='right')
plt.ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
plt.title('Feature Correlation with Species', fontsize=14, fontweight='bold', pad=20)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

for i, (bar, value) in enumerate(zip(bars, correlations.values)):
    plt.text(i, value + 0.02 if value > 0 else value - 0.05, 
             f'{value:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/feature_correlation.png', dpi=300, bbox_inches='tight')
print("\n* Saved: visualizations/feature_correlation.png")

# Summary
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

print("\nFeature Rankings:")
combined_ranks = pd.DataFrame({
    'RF_Importance': importance_df.set_index('Feature')['Importance'].rank(ascending=False),
    'Perm_Importance': perm_importance_df.set_index('Feature')['Importance'].rank(ascending=False),
    'Correlation': correlations.abs().rank(ascending=False)
})
combined_ranks['Average_Rank'] = combined_ranks.mean(axis=1)
combined_ranks = combined_ranks.sort_values('Average_Rank')

print(f"\n{'Feature':<25} {'RF':>5} {'Perm':>5} {'Corr':>5} {'Avg':>7}")
print("-"*70)
for feature, row in combined_ranks.iterrows():
    print(f"{feature:<25} {row['RF_Importance']:>5.1f} {row['Perm_Importance']:>5.1f} "
          f"{row['Correlation']:>5.1f} {row['Average_Rank']:>7.2f}")

top_feature = combined_ranks.index[0]
print(f"\n MOST IMPORTANT FEATURE: {top_feature}")
print(f"\n* Petal measurements are most discriminative!")
print("\n* Feature importance analysis completed!")