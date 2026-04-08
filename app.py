import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import os
from tensorflow.keras.models import load_model

# 🌸 Streamlit App Configuration
st.set_page_config(
    page_title="Flower Recognition App",
    page_icon="🌼",
    layout="centered"
)

# 🌺 Background Styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://c4.wallpaperflare.com/wallpaper/295/401/83/flowers-background-dark-patterns-wallpaper-preview.jpg");
        background-size: cover;
        background-position: center;
    }
    .info-card {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        padding: 20px;
        margin-top: 15px;
        color: #2b2b2b;
        font-size: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .title-text {
        text-align: center;
        font-size: 40px;
        color: white;
        text-shadow: 2px 2px 5px #000000;
    }
    .sub-text {
        text-align: center;
        color: #f0f0f0;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 🌷 Title
st.markdown("<h1 class='title-text'>🌼 Flower Recognition App</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Identify flowers using Deep Learning and learn their botanical details</p>", unsafe_allow_html=True)

# 🌻 Load the Trained CNN Model
model = load_model('Flower_Recognition_Model.keras')

# Flower labels
flowers_name = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# 📘 Flower Information Dictionary
flower_info = {
    "daisy": """
    <b>🌸 Scientific Name:</b> <i>Bellis perennis</i><br>
    <b>🌿 Family:</b> Asteraceae<br>
    <b>📖 Description:</b> Small white petals with yellow centers, symbolizing purity and innocence.<br>
    <b>🏞️ Habitat:</b> Found in meadows and lawns across Europe and North America.<br>
    <b>💡 Fun Fact:</b> Daisies close their petals at night and reopen with sunlight!
    """,
    "dandelion": """
    <b>🌼 Scientific Name:</b> <i>Taraxacum officinale</i><br>
    <b>🌿 Family:</b> Asteraceae<br>
    <b>📖 Description:</b> Yellow flowers that form fluffy seed heads dispersed by the wind.<br>
    <b>🏞️ Habitat:</b> Common in temperate regions worldwide.<br>
    <b>💡 Fun Fact:</b> Every part of the dandelion (root, leaves, flower) is edible!
    """,
    "rose": """
    <b>🌹 Scientific Name:</b> <i>Rosa spp.</i><br>
    <b>🌿 Family:</b> Rosaceae<br>
    <b>📖 Description:</b> Fragrant flowers available in many colors, symbolizing love and beauty.<br>
    <b>🏞️ Habitat:</b> Grown globally, in gardens and wild regions.<br>
    <b>💡 Fun Fact:</b> There are over 300 species and thousands of varieties of roses!
    """,
    "sunflower": """
    <b>🌻 Scientific Name:</b> <i>Helianthus annuus</i><br>
    <b>🌿 Family:</b> Asteraceae<br>
    <b>📖 Description:</b> Tall, bright yellow flowers that follow the sun (heliotropism).<br>
    <b>🏞️ Habitat:</b> Native to North America; cultivated for oil and seeds.<br>
    <b>💡 Fun Fact:</b> Sunflowers can remove toxins from soil (phytoremediation)!
    """,
    "tulip": """
    <b>🌷 Scientific Name:</b> <i>Tulipa spp.</i><br>
    <b>🌿 Family:</b> Liliaceae<br>
    <b>📖 Description:</b> Cup-shaped colorful flowers that bloom in spring.<br>
    <b>🏞️ Habitat:</b> Native to Central Asia, widely grown in the Netherlands.<br>
    <b>💡 Fun Fact:</b> Tulips were once more valuable than gold during the Tulip Mania era!
    """
}

# 🧠 Function: Classify Uploaded Image
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    flower = flowers_name[np.argmax(result)]
    confidence = np.max(result) * 100
    return flower, confidence

# 🌸 Function: Fetch Flower Images (from Unsplash)
def get_flower_images(flower_name, base_path="dataset"):
    flower_dir = os.path.join(base_path, flower_name.lower())
    images = []

    # Check if folder exists
    if not os.path.exists(flower_dir):
        st.warning(f"No local images found for '{flower_name}'.")
        return images

    # Read image files
    for file_name in os.listdir(flower_dir):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(flower_dir, file_name)
            try:
                img = Image.open(img_path)
                images.append(img)
            except Exception as e:
                st.warning(f"⚠️ Could not load {file_name}: {e}")
    return images

 


# 🌼 Tabs: Search | Upload
tab1, tab2 = st.tabs(["🔍 Search Flower Info", "📸 Upload & Identify"])

# 🔍 Search Tab
# 🔍 Search Tab
with tab1:
    st.subheader("Search for a Flower")
    search_flower = st.text_input("Enter flower name (e.g., rose, tulip, sunflower):")

    if search_flower:
        flower_key = search_flower.strip().lower()
        if flower_key in flower_info:
            st.markdown(f"<div class='info-card'>{flower_info[flower_key]}</div>", unsafe_allow_html=True)
            st.markdown("### 📷 Local Dataset Images:")

            images = get_flower_images(flower_key)
            if images:
                # Limit number of images shown
                images = images[:5]
                cols = st.columns(len(images))
                for i, img in enumerate(images):
                    with cols[i]:
                        st.image(img, use_container_width=True)
            else:
                st.info("No images found in your dataset folder.")
        else:
            st.warning("No information available for that flower.")
    else:
        st.info("Type a flower name to search its info and view images.")

# 📤 Upload Tab
with tab2:
    st.subheader("Upload an Image for Recognition")
    upload_file = st.file_uploader('Upload a flower image', type=["jpg", "png", "jpeg"])

    if upload_file is not None:
        if not os.path.exists('upload'):
            os.makedirs('upload')
        file_path = os.path.join('upload', upload_file.name)
        with open(file_path, 'wb') as f:
            f.write(upload_file.getbuffer())

        st.image(upload_file, width=300)
        flower, confidence = classify_images(file_path)

        st.markdown(f"<h3 style='color:white;'>Prediction: {flower.title()} 🌼</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:white;'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)

        if flower in flower_info:
            st.markdown(f"<div class='info-card'>{flower_info[flower]}</div>", unsafe_allow_html=True)
        else:
            st.info("No additional botanical details available for this flower.")
    else:
        st.info("Please upload an image to classify.")
