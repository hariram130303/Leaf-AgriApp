import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Caching model load
@st.cache_resource
def load_keras_model():
    return load_model("Model.keras")

model = load_keras_model()

# Class labels
labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]  # (keep your original list here)

def model_predict(img, model):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    predicted_index = np.argmax(preds)
    predicted_probability = preds[0][predicted_index]
    class_name = labels[predicted_index].split("___")
    return class_name, predicted_probability

# Set page config
st.set_page_config(page_title="Plant Disease Classifier", layout="centered")

# Sidebar for navigation
st.sidebar.title("🌿 Navigation")
page = st.sidebar.radio("Go to", ["Home", "How It Works", "Visualizations", "About", "Feedback"])

# --- Page 1: Home ---
if page == "Home":
    st.title("🌿 Plant Disease Classifier")
    st.markdown("Upload a leaf image to detect the crop and its disease using a deep learning model.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Leaf Image', use_column_width=True)

        with st.spinner('Predicting...'):
            (crop_disease, probability) = model_predict(img, model)
            crop, disease = crop_disease
            st.success(f"**Predicted Crop:** {crop.replace('_', ' ')}")
            st.info(f"**Predicted Disease:** {disease.replace('_', ' ').title()}")
            st.metric(label="Prediction Confidence", value=f"{probability * 100:.2f}%")

# --- Page 2: How It Works ---
elif page == "How It Works":
    st.title("🔍 How It Works")
    st.markdown("""
    - This system uses a Convolutional Neural Network (CNN) trained on the **PlantVillage** dataset.
    - Images are resized to 224x224 and normalized before feeding into the model.
    - The model predicts the crop type and disease using deep learning classification.
    """)

    # st.image("workflow.png", caption="Model Workflow", use_column_width=True)

# --- Page 3: Visualizations (Placeholder) ---
elif page == "Visualizations":
    st.title("🧠 Model Visualizations")
    st.markdown("Below are a few visual insights related to the dataset and model predictions.")

    # --- Simulated Class Distribution Bar Chart ---
    st.subheader("1️⃣ Class Distribution (Simulated)")
    import matplotlib.pyplot as plt
    import pandas as pd

    class_counts = {
        'Tomato___Early_blight': 400,
        'Potato___Late_blight': 380,
        'Tomato___healthy': 350,
        'Corn___Common_rust': 300,
        'Apple___Black_rot': 250
    }

    df_counts = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
    st.bar_chart(df_counts.set_index('Class'))

    # --- Simulated Confidence Distribution ---
    st.subheader("2️⃣ Sample Confidence Distribution")
    confidences = np.random.normal(loc=0.85, scale=0.1, size=100)
    confidences = np.clip(confidences, 0, 1)

    fig, ax = plt.subplots()
    ax.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Prediction Confidence Distribution")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


    # --- Optional: Upload an image to show prediction distribution ---
    st.subheader("4️⃣ Try a New Image (Optional)")
    uploaded_vis = st.file_uploader("Upload an image for detailed prediction", type=["jpg", "jpeg", "png"], key="visuploader")

    if uploaded_vis is not None:
        img = Image.open(uploaded_vis)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        img_resized = img.resize((224, 224))
        x = image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0) / 255.0
        preds = model.predict(x)[0]

        top_indices = np.argsort(preds)[::-1][:5]
        top_preds = [(labels[i], preds[i]) for i in top_indices]

        pred_df = pd.DataFrame(top_preds, columns=["Class", "Confidence"])
        st.bar_chart(pred_df.set_index("Class"))
        st.markdown("**Top 5 Predictions:**")
        for cls, conf in top_preds:
            st.write(f"{cls.replace('_', ' ').title()}: {conf:.2f}")
# --- Page 4: About ---
elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    - **Project**: Plant Disease Detection using Deep Learning
    - **Dataset**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
    - **Framework**: TensorFlow + Keras + Streamlit
    - **Author**: HARIRAM THOGATA MADAM
    """)

# --- Page 5: Feedback ---
elif page == "Feedback":
    st.title("💬 Feedback Form")
    st.markdown("We appreciate your feedback to improve the app!")

    # Collect user input
    name = st.text_input("Your Name", max_chars=50)
    rating = st.slider("Rate the App", 1, 5, 3)
    feedback = st.text_area("Share your thoughts or suggestions...", height=150)

    # Submit button
    if st.button("Submit Feedback"):
        if name and feedback:
            import pandas as pd
            from datetime import datetime
            feedback_data = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Name": name,
                "Rating": rating,
                "Feedback": feedback
            }

            # Save to CSV
            feedback_file = "feedback.csv"
            if os.path.exists(feedback_file):
                df = pd.read_csv(feedback_file)
                df = df.append(feedback_data, ignore_index=True)
            else:
                df = pd.DataFrame([feedback_data])

            df.to_csv(feedback_file, index=False)

            st.success("✅ Thank you! Your feedback has been recorded.")
        else:
            st.warning("⚠️ Please enter your name and feedback.")


