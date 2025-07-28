import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import tempfile

# UI config
st.set_page_config(page_title="🎯 LSTM Predictor", layout="centered")
st.title("🔮 Predict Next Parity Result")

# Upload .h5 model
uploaded_model = st.file_uploader("", type=["h5"])
model = None

if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(uploaded_model.read())
        model_path = tmp.name
    model = load_model(model_path)
    st.success("✅ Model loaded successfully!")

# If model is loaded, accept input
if model:
    label_map = {0: 'Green/Violet', 1: 'Red'}
    st.subheader("🔢 Enter last 10 numbers (last digit of price):")
    user_input = st.text_input("Example: 0,9,0,3,3,3,7,0,7,4")

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
                result = label_map[predicted_class]
                st.markdown(f"### 🎯 Prediction: **{result}**")
                st.markdown(f"🧠 Confidence: `{confidence * 100:.2f}%`")

                if confidence < 0.6:
                    st.warning("⚠️ Low confidence — WAIT recommended.")
                elif confidence >= 0.8:
                    st.success("✅ High confidence — Consider acting!")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
else:
    st.info("👆 Please upload a `.h5` LSTM model to start prediction.")
