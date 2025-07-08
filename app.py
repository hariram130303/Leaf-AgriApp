# app.py

import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image as keras_image

# Set page config
st.set_page_config(page_title="Plant Disease Classifier", layout="centered")

# Load model once and cache it
@st.cache_resource
def load_keras_model():
    model = load_model("Model.h5")
    model.make_predict_function()
    return model

model = load_keras_model()

# Class labels
CLASS_LABELS = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 
'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 
'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


def predict_class(img, model):
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    preds = model.predict(img_array)[0]
    class_index = np.argmax(preds)
    label = CLASS_LABELS[class_index]
    crop, disease = label.split('___')
    return crop, disease.replace('_', ' ').title(), preds[class_index]

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "Model Info", "About Project", "Contact"])

# --------------------- Home Page ---------------------
if app_mode == "Home":
    st.title("🌿 Plant Disease Classifier")
    st.markdown("Upload a leaf image to detect the crop and disease.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_data = Image.open(uploaded_file)
        st.image(image_data, caption="Uploaded Leaf Image", use_column_width=True)

        with st.spinner("Predicting..."):
            crop, disease, confidence = predict_class(image_data, model)

        st.success(f"🌾 **Crop:** {crop}")
        st.error(f"🦠 **Disease:** {disease}")
        st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")

# --------------------- Model Info Page ---------------------
elif app_mode == "Model Info":
    st.title("📊 Model Information")
    st.markdown("""
    - **Architecture:** Custom CNN inspired by AlexNet  
    - **Input Size:** 224 x 224 x 3  
    - **Framework:** Keras (TensorFlow backend)  
    - **Optimizer:** Adam  
    - **Loss Function:** Sparse Categorical Crossentropy  
    - **Output:** 38 Plant Disease Classes
    """)

    if st.button("Show Model Summary in Terminal"):
        model.summary(print_fn=st.write)

# --------------------- About Project Page ---------------------
elif app_mode == "About Project":
    st.title("📘 About This Project")
    st.markdown("""
    This AI-powered web application detects plant diseases from leaf images.  
    It supports 38 classes from various crops including:
    
    - **Apple**, **Corn**, **Tomato**, **Grape**, **Potato**, **Pepper**, etc.  
    - Diseases include **blight**, **rot**, **mold**, **rust**, and more.
    
    **Key Goals:**
    - Early disease detection for farmers
    - Reduced crop loss
    - Accessible diagnosis via web/mobile
    """)

# --------------------- Contact Page ---------------------
elif app_mode == "Contact":
    st.title("📬 Contact Developer")
    st.markdown("""
    - **Name:** Hari Ram  
    - 📧 Email: tmhariram@gmail.com  
    - 💼 [LinkedIn](https://linkedin.com/in/hari-ram-thogata-madam)  
    - 💻 [GitHub](https://github.com/hariram130303)  
    """)

