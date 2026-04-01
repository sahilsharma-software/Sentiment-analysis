import streamlit as st
import cv2
import numpy as np
import os
import time
from keras.models import load_model
from youtubesearchpython import VideosSearch
from PIL import Image

# === Load trained model ===
model = load_model("emotion_model.h5")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# === Emotion to song mapping ===
emotion_song_themes = {
    'angry': "Sidhu Moose Wala songs",
    'happy': "Nonstop Hindi Mashup 2024",
    'sad': "Arijit Singh and Rahat Fateh Ali Khan emotional songs",
    'neutral': "Karan Aujla chill Punjabi songs",
    'surprise': "Mind-blowing trending Indian reels songs 2024",
    'disgust': "Aafat type viral Bollywood songs",
    'fear': "Calm emotional background music",
}

# === Create screenshots folder ===
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# === Streamlit Web App ===
st.set_page_config(page_title="Emotion Music Recommender", layout="centered")
st.title("🎵 Emotion-Based Music Recommender")
st.markdown("Capture your face and get a matching **Gen-Z song**!")

# === Webcam input ===
img_data = st.camera_input("Click a selfie to detect your emotion 👇")

if img_data:
    # Convert webcam image to grayscale
    img = Image.open(img_data)
    gray_img = img.convert('L')  # Grayscale
    img_array = np.array(gray_img)

    # Preprocess for model
    face = cv2.resize(img_array, (48, 48))
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))

    # Predict emotion
    prediction = model.predict(face, verbose=0)
    emotion = emotion_labels[np.argmax(prediction)]

    # Show emotion
    st.success(f"🧠 Detected Emotion: **{emotion.upper()}**")

    # Save screenshot if angry or fear
    if emotion in ['angry', 'fear']:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"screenshots/{emotion}_{timestamp}.png"
        img.save(screenshot_path)
        st.warning(f"📸 Screenshot saved: `{screenshot_path}`")

    # === Show Top 3 YouTube Song Recommendations ===
    search_query = emotion_song_themes.get(emotion)
    if search_query:
        st.markdown("### 🎧 Top 3 Song Recommendations:")
        with st.spinner("🔍 Fetching songs..."):
            try:
                results = VideosSearch(search_query, limit=3).result()
                for video in results['result']:
                    title = video['title']
                    video_url = video['link']
                    thumbnail = video['thumbnails'][0]['url']

                    with st.container():
                        st.image(thumbnail, width=320)
                        st.write(f"**{title}**")
                        st.markdown(f"[▶️ Watch on YouTube]({video_url})", unsafe_allow_html=True)
                        st.markdown("---")
            except Exception as e:
                st.error(f"⚠️ Could not load videos: {e}")
