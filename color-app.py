import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import tempfile

st.set_page_config(page_title="🎯 LSTM Predictor", layout="centered")
st.title("🔮 Predict Next Parity Result")

# File uploader
uploaded_model = st.file_uploader("📤 Upload your trained .h5 LSTM model", type=["h5"])
model = None

if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(uploaded_model.read())
        tmp_path = tmp.name
    model = load_model(tmp_path)
    st.success("✅ Model loaded successfully!")

# Prediction interface
if model:
    label_map = {0: 'Green/Violet', 1: 'Red'}
    st.subheader("🎯 Enter the last 10 numbers (last digit of price):")
    user_input = st.text_input("🔢 Comma-separated digits (e.g., 3,2,1,0,9,3,7,1,6,4)")

    if st.button("Predict"):
        try:
            last_10 = [int(x.strip()) for x in user_input.split(',')]
            if len(last_10) != 10:
                st.error("❌ You must enter exactly 10 digits.")
            else:
                input_array = np.array(last_10).reshape((1, 10, 1))
                prediction = model.predict(input_array)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                st.markdown(f"### 🎯 Prediction: **{label_map[predicted_class]}**")
                st.markdown(f"Confidence: `{confidence * 100:.2f}%`")
                if confidence < 0.6:
                    st.warning("⚠️ Low confidence — WAIT recommended!")
                elif confidence > 0.8:
                    st.success("✅ High confidence — You may consider acting!")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("👆 Please upload a trained LSTM model (.h5) to begin.")
