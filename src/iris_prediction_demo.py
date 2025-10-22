"""
IRIS PREDICTION DEMO
Use the trained model to classify new iris flowers
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load and prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train best model (SVM)
model = SVC(kernel='rbf', C=10, gamma='scale')
model.fit(X_train_scaled, y_train)

print("="*80)
print("IRIS FLOWER CLASSIFIER - PREDICTION DEMO")
print("="*80)

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict iris species from measurements
    
    Parameters:
    -----------
    sepal_length : float (cm)
    sepal_width  : float (cm)
    petal_length : float (cm)
    petal_width  : float (cm)
    """
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    species = iris.target_names[prediction]
    
    return species

# Example predictions
print("\n" + "-"*80)
print("EXAMPLE PREDICTIONS")
print("-"*80)

examples = [
    {
        'name': 'Small flower',
        'measurements': (5.0, 3.5, 1.3, 0.3),
        'expected': 'Setosa'
    },
    {
        'name': 'Medium flower',
        'measurements': (6.0, 2.9, 4.5, 1.5),
        'expected': 'Versicolor'
    },
    {
        'name': 'Large flower',
        'measurements': (6.5, 3.0, 5.8, 2.2),
        'expected': 'Virginica'
    }
]

for example in examples:
    sl, sw, pl, pw = example['measurements']
    species = predict_iris(sl, sw, pl, pw)
    
    print(f"\n{example['name']}:")
    print(f"  Sepal: {sl} cm × {sw} cm")
    print(f"  Petal: {pl} cm × {pw} cm")
    print(f"  → Predicted: {species}")
    print(f"  → Expected:  {example['expected']}")
    print(f"  → Match: {'*' if species == example['expected'] else '✗'}")

# Custom prediction
print("\n" + "="*80)
print("CUSTOM PREDICTION EXAMPLE")
print("="*80)

test_measurements = (5.8, 3.1, 5.0, 1.8)
sl, sw, pl, pw = test_measurements

print(f"\nFlower measurements:")
print(f"  Sepal Length: {sl} cm")
print(f"  Sepal Width:  {sw} cm")
print(f"  Petal Length: {pl} cm")
print(f"  Petal Width:  {pw} cm")

species = predict_iris(sl, sw, pl, pw)

print(f"\n{'='*80}")
print(f"PREDICTION: {species.upper()}")
print(f"{'='*80}")

print("\nModel accuracy on test set:", model.score(X_test_scaled, y_test))
print("\n* Demo completed successfully!")