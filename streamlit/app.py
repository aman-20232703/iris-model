import streamlit as st
import numpy as np
import pickle

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="🌸 Iris Flower Classifier",
    page_icon="🌼",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ------------------- CUSTOM CSS -------------------
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%);
    font-family: 'Poppins', sans-serif;
}
.css-18e3th9 {
    padding-top: 2rem;
}
h1 {
    color: #6a0dad;
    text-align: center;
    font-weight: 800;
    text-shadow: 1px 1px 2px #fff;
}
div[data-baseweb="slider"] span {
    color: #6a0dad !important;
}
.stButton>button {
    background-color: #6a0dad;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-weight: bold;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #8b3cfb;
    transform: scale(1.03);
}
.result-box {
    background-color: #ffffffcc;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ------------------- LOAD MODEL -------------------
import os
import pickle

# Get the folder where this script (app.py) resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct full path to the model
MODEL_PATH = os.path.join(BASE_DIR, "iris_svm_model.pkl")

# Load the model safely
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
else:
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    model = data['model']
    scaler = data['scaler']
    encoder = data['encoder']

# ------------------- HEADER -------------------
st.title("🌸 Iris Flower Prediction (SVM Model)")
st.markdown("### Predict the type of Iris flower by adjusting the sliders below 👇")

# ------------------- SLIDER INPUTS -------------------
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("🌿 Sepal Length (cm)", 4.0, 8.0, 5.8)
    petal_length = st.slider("🌼 Petal Length (cm)", 1.0, 7.0, 4.2)
with col2:
    sepal_width  = st.slider("🌿 Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_width  = st.slider("🌼 Petal Width (cm)", 0.0, 2.5, 1.3)

# ------------------- PREDICTION -------------------
if st.button("🔮 Predict Flower Type"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)
    species = encoder.inverse_transform(pred)[0]

    st.markdown(f"""
    <div class="result-box">
        <h2>🌷 Predicted Iris Species:</h2>
        <h1 style='color:#6a0dad;'>{species}</h1>
        <p style='color:#555;'>Based on the given petal and sepal measurements.</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------- SIDEBAR INFO -------------------
st.sidebar.header("📘 About This App")
st.sidebar.info("""
This **Iris Flower Prediction App** uses a **Support Vector Machine (SVM)** model 
trained on the classic Iris dataset 🌸.  
It helps you identify whether a flower is:
- *Iris Setosa*  
- *Iris Versicolor*  
- *Iris Virginica*  
Just move the sliders and click Predict!
""")

st.sidebar.write("💻 Built with Streamlit, NumPy, and Scikit-learn")
