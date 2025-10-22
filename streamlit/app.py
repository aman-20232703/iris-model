import streamlit as st
import numpy as np
import pickle

# Load everything back
with open("iris_svm_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data['model']
scaler = data['scaler']
encoder = data['encoder']

st.title("🌸 Iris Flower Prediction (SVM Model)")

# User inputs
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.2)
petal_width  = st.slider("Petal Width (cm)", 0.0, 2.5, 1.3)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)
    species = encoder.inverse_transform(pred)[0]
    st.success(f"Predicted Iris Species: 🌼 {species}")
