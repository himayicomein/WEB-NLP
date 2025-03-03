import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import time
from googletrans import Translator
import datetime  

# โหลดโมเดล
loaded_model = keras.models.load_model("Model3/hate_speech_modelv3.h5")

with open("Model3/label_encoder.pkl", "rb") as f:
    loaded_label_encoder = pickle.load(f)

with open("Model3/tokenizer.pkl", "rb") as f:
    loaded_tokenizer = pickle.load(f)

# Mapping ค่าที่จำแนก
class_mapping = {
    0: "🚨 Hate Speech",
    1: "⚠️ Offensive Language",
    2: "✅ Neither"
}

advice_mapping = {
    "🚨 Hate Speech": "โปรดหลีกเลี่ยงการใช้ถ้อยคำที่แสดงความเกลียดชัง และพยายามใช้ภาษาที่สร้างสรรค์",
    "⚠️ Offensive Language": "ข้อความของคุณอาจมีเนื้อหาที่ไม่เหมาะสม โปรดพิจารณาปรับเปลี่ยนคำพูดให้สุภาพขึ้น"
}

# โหลดประวัติอินพุต
try:
    with open("history.json", "r", encoding="utf-8") as f:
        input_history = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    input_history = []

# ฟังก์ชันทำนาย
def predict_text(text):
    translator = Translator()
    translated_text = translator.translate(text, src="auto", dest="en").text
    st.write(f"📝 Translated Text: {translated_text}")
    
    sequence = loaded_tokenizer.texts_to_sequences([translated_text])
    padded = pad_sequences(sequence, maxlen=100, padding="post", truncating="post")
    prediction = loaded_model.predict(padded)
    predicted_class = np.argmax(prediction)
    result = class_mapping.get(predicted_class, "Unknown")
    
    # บันทึกเวลา
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    input_data = {
        "original_text": text,
        "translated_text": translated_text,
        "prediction": result,
        "time": current_time
    }
    input_history.append(input_data)
    with open("history.json", "w", encoding="utf-8") as f:
        json.dump(input_history, f, indent=4, ensure_ascii=False)
    
    return result

# ตั้งค่า Streamlit
st.set_page_config(page_title="Hate Speech Detector", page_icon="🚨", layout="wide")

# สร้าง Sidebar สำหรับนำทาง
st.sidebar.title("🔍 Hate Speech Detector")
page = st.sidebar.radio("เลือกเมนู", ["📝 Test", "📜 History"])

# **หน้าที่ 1: ทดสอบข้อความ**
if page == "📝 Test":
    st.markdown("<h1 style='text-align:center; color:#FF4B4B;'>⚠️ Hate Speech Detector</h1>", unsafe_allow_html=True)
    st.markdown("<h3>📝 Input your text below:</h3>", unsafe_allow_html=True)
    
    user_input = st.text_area("💬 Enter your text:", height=150)
    
    if st.button("🔍 Analyze Text"):
        if user_input.strip():
            prediction = predict_text(user_input)
            
            # สร้าง Progress Bar
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress.progress(i + 1)

            if "Hate Speech" in prediction:
                st.error(f"{prediction}")
                st.warning(f"💡 Suggestion: {advice_mapping[prediction]}")
            elif "Offensive Language" in prediction:
                st.warning(f"{prediction}")
                st.info(f"💡 Suggestion: {advice_mapping[prediction]}")
            else:
                st.success(f"{prediction}")
                st.balloons()
        else:
            st.warning("⚠️ กรุณาใส่ข้อความก่อนกดปุ่มวิเคราะห์!")

# **หน้าที่ 2: ประวัติการตรวจสอบ**
elif page == "📜 History":
    st.markdown("<h1 style='text-align:center; color:#4BA3FF;'>📜 ประวัติการตรวจสอบ</h1>", unsafe_allow_html=True)
    
    if input_history:
        for idx, entry in enumerate(reversed(input_history[-10:]), 1):  # แสดงรายการล่าสุด 10 รายการ
            with st.expander(f"🔹 ประวัติการค้นหา {idx}"):
                st.write(f"**⏰ เวลา:** {entry.get('time', 'N/A')}")
                st.write(f"**📝 Original Text:** {entry['original_text']}")
                st.write(f"**🔁 Translated Text:** {entry['translated_text']}")
                st.write(f"**🔍 Prediction:** {entry['prediction']}")
    else:
        st.info("❌ ยังไม่มีประวัติการตรวจสอบข้อความ")
