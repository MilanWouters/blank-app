import streamlit as st
import cv2
import numpy as np
import tempfile
import tensorflow.keras.models as tf
from PIL import Image, ImageOps
import pandas as pd
import re
import io  # Importeer io voor BytesIO

# --- Page Config ---
st.set_page_config(page_title="AI Dieren Classifier 🦁", page_icon="🤖", layout="wide")


# --- Label opschoning functie ---
def clean_label(label):
    return re.sub(r"^\d+\s*", "", label).strip()


# --- Dierenlijst met emoji's ---
animal_emoji = {
    "hond": "🐶", "kat": "🐱", "tijger": "🐯", "koe": "🐮", "paard": "🐴",
    "varken": "🐷", "schaap": "🐑", "geit": "🐐", "kip": "🐔", "eend": "🦆",
    "duif": "🕊", "olifant": "🐘", "leeuw": "🦁", "aap": "🐵", "beer": "🐻",
    "wolf": "🐺", "zebra": "🦓", "pinguin": "🐧", "nijlpaard": "🦛", "mens": "🧑"
}


def get_animal_emoji(label):
    sorted_animals = sorted(animal_emoji.keys(), key=len, reverse=True)  # Sorteer op lengte
    for animal in sorted_animals:
        if animal in label.lower():
            return f"{animal_emoji[animal]} {label}"
    return f"❓ {label}"


# --- Model laden ---
@st.cache_resource
def load_ai_model():
    return tf.load_model("keras_model.h5", compile=False)


model = load_ai_model()
class_names = [clean_label(line) for line in open("labels.txt", "r").readlines()]

# --- Sidebar: Kies modus ---
st.sidebar.title("📌 Kies een modus")
option = st.sidebar.radio("Wat wil je analyseren?", ["📷 Afbeelding Classificatie", "🎥 Video Classificatie"])

# --- **Afbeelding Classificatie** ---
if option == "📷 Afbeelding Classificatie":
    st.title("🔍 AI Dieren Classifier")
    st.subheader("Laat een AI-model een afbeelding herkennen!")

    uploaded_image = st.file_uploader("📸 Upload een afbeelding", type=["jpg", "png", "jpeg"])
    image = st.camera_input("📷 Maak een foto")

    processed_image = None
    show_top3 = False

    if uploaded_image:
        try:
            processed_image = Image.open(io.BytesIO(uploaded_image.getvalue()))  # Fix voor upload
            show_top3 = False  # Normale weergave voor geüploade afbeelding
        except Exception as e:
            st.error(f"❌ Er is een fout opgetreden bij het openen van de afbeelding: {e}")

    elif image:
        try:
            processed_image = Image.open(io.BytesIO(image.getvalue()))  # Fix voor camera-input
            show_top3 = True  # Camera-input toont top 3
        except Exception as e:
            st.error(f"❌ Er is een fout opgetreden bij het verwerken van de foto: {e}")

    if processed_image:
        with st.spinner("🔄 AI is bezig met analyseren..."):
            size = (224, 224)
            processed_image = ImageOps.fit(processed_image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(processed_image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            prediction = model.predict(data)[0]
            top_indices = np.argsort(prediction)[-3:][::-1]  # Top 3 scores
            top_labels = [class_names[i] for i in top_indices]
            top_scores = [prediction[i] * 100 for i in top_indices]
            best_label = top_labels[0]
            best_category = get_animal_emoji(best_label)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(processed_image, caption="📷 Ingevoerde afbeelding", use_container_width=True)
            img_byte_arr = tempfile.NamedTemporaryFile(delete=False)
            processed_image.save(img_byte_arr, format="PNG")
            st.download_button("📥 Download afbeelding", img_byte_arr.name, "image.png", "image/png")

        with col2:
            st.subheader("📊 AI Resultaat")
            if show_top3:
                # Top 3 resultaten in een tabel weergeven zonder index en met "zekerheid"
                df = pd.DataFrame({
                    "Dier": [get_animal_emoji(label) for label in top_labels],
                    "Zekerheid (%)": [f"{score:.2f}%" for score in top_scores]
                })
                st.dataframe(df, hide_index=True, use_container_width=True)
                st.markdown(f"### 📌 Meest waarschijnlijk: {best_category}")
            else:
                # Normale weergave als het een geüploade afbeelding is
                st.markdown(f"### {best_category} met **{top_scores[0]:.2f}%** zekerheid")

# --- **Video Classificatie (Top 1)** ---
elif option == "🎥 Video Classificatie":
    st.title("🎥 AI Dieren Classifier voor Video's")
    uploaded_video = st.file_uploader("📹 Upload een video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name

        st.video(video_path)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        st.write(f"📌 Frames: {frame_count}, FPS: {frame_rate}")

        predictions = {}
        with st.spinner("🔄 AI is bezig met analyseren..."):
            frame_step = max(1, frame_rate // 2)
            while cap.isOpened():
                frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % frame_step == 0:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                    image_array = np.asarray(image)
                    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                    data[0] = normalized_image_array

                    prediction = model.predict(data)[0]
                    best_index = np.argmax(prediction)
                    best_label = class_names[best_index]
                    predictions[best_label] = predictions.get(best_label, 0) + 1

            cap.release()

        best_label = max(predictions, key=predictions.get)
        best_category = get_animal_emoji(best_label)

        st.subheader("📊 AI Resultaat")
        st.markdown(f"### Het herkende dier is: **{best_category}** 🎯")

# --- Footer ---
st.markdown("---")
st.write("💡 **Gemaakt met Streamlit, TensorFlow & Keras!**")
st.write("📢 **Credits:** Milan Wouters, Raoul Buffels, Jorre Van Dyck en Seppe Van Roy")