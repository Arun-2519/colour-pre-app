# app.py
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tempfile
import datetime

st.set_page_config(page_title="🎯 LSTM Predictor", layout="centered")
st.title("🔮 Parity Prediction (LSTM-Based)")

if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "correct" not in st.session_state:
    st.session_state.correct = 0
if "total" not in st.session_state:
    st.session_state.total = 0

uploaded_model = st.file_uploader("📤 Upload `.h5` LSTM model", type=["h5"])
model = None

if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(uploaded_model.read())
        model_path = tmp.name
    model = load_model(model_path)
    st.success("✅ Model loaded successfully!")

def export_to_excel():
    df = pd.DataFrame(st.session_state.predictions)
    file_name = f"predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df.to_excel(file_name, index=False)
    return file_name

def play_sound(type):
    if type == "wait":
        st.audio("https://www.soundjay.com/button/sounds/beep-07.mp3")
    elif type == "high":
        st.audio("https://www.soundjay.com/button/sounds/beep-10.mp3")
    elif type == "normal":
        st.audio("https://www.soundjay.com/button/sounds/beep-01a.mp3")

if model:
    label_map = {0: 'Green/Violet', 1: 'Red'}
    st.subheader("🔢 Enter last 10 numbers:")
    user_input = st.text_input("Example: 0,9,0,3,3,3,7,0,7,4")
    actual_result = st.selectbox("What was the actual result?", ["", "Red", "Green/Violet"])

    if st.button("🔮 Predict"):
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
                    play_sound("wait")
                    st.warning("⚠️ WAIT — Low confidence.")
                elif confidence >= 0.8:
                    play_sound("high")
                    st.success("✅ High confidence — Go ahead!")
                else:
                    play_sound("normal")

                if actual_result != "":
                    actual_label = 1 if actual_result == "Red" else 0
                    correct = (actual_label == predicted_class)
                    st.session_state.total += 1
                    if correct:
                        st.session_state.correct += 1
                    st.session_state.predictions.append({
                        "Input": user_input,
                        "Prediction": result,
                        "Confidence": round(confidence * 100, 2),
                        "Actual": actual_result,
                        "Correct": correct
                    })

        except Exception as e:
            st.error(f"❌ Error: {e}")

    # Show accuracy stats
    if st.session_state.total > 0:
        acc = (st.session_state.correct / st.session_state.total) * 100
        st.markdown(f"### 📈 Accuracy: `{acc:.2f}%`")

    if st.session_state.predictions:
        st.markdown("### 🧾 Prediction History")
        st.dataframe(pd.DataFrame(st.session_state.predictions))

        if st.button("📥 Export to Excel"):
            file_path = export_to_excel()
            with open(file_path, "rb") as f:
                st.download_button("📤 Download Excel", f, file_name=file_path)
else:
    st.info("👆 Please upload your `.h5` model to start.")
