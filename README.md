# Leaf-AgriApp – Crop Disease Detection

> **Live App**: [Deployed on Heroku](#) (replace with actual link)

A web application that uses deep learning to classify and detect diseases in crop leaves. Built using CNN and Flask, and trained on a dataset of 15 crops with 38 disease classes.

---

## 📑 Table of Contents

- [Project Title](#project-title)
- [Problem Statement](#problem-statement)
- [Project Description](#project-description)
  - [Image Acquisition](#image-acquisition)
  - [Image Pre-processing](#image-pre-processing)
  - [Feature Extraction](#feature-extraction-and-selection)
  - [Model Training](#model-training)
  - [Image Classification](#image-classification)
  - [Web App using Flask](#web-app-using-flask)
  - [Model Deployment](#model-deployment)
- [Assumptions](#assumptions)
- [Technologies & Tools](#technologies--tools)
- [Advantages](#advantages)
- [Future Scope](#future-scope)
- [Conclusion](#conclusion)
- [Dataset](#dataset)

---

## 🎯 Project Title

In agriculture, leaf diseases cause significant losses in crop yield and quality. This project aims to automate the recognition of leaf diseases using deep learning and a CNN model.

---

## ❗ Problem Statement

The agricultural sector faces challenges due to undetected plant diseases, especially in regions where farming is a primary livelihood. Early detection can prevent major crop losses. This project helps detect such diseases using images of plant leaves.

---

## 📚 Project Description

The system uses a Convolutional Neural Network (CNN) based deep learning model to classify leaf images. Here's how it works:

### 📷 Image Acquisition
- Images are captured via camera.
- White backgrounds are preferred for better segmentation.

### 🧼 Image Pre-processing
- High-resolution images are resized.
- Batch Normalization is applied for normalization.

### 🧠 Feature Extraction and Selection
- CNN layers extract features automatically.
- MaxPooling and Dropout are used to select the most relevant ones.

### 🏋️ Model Training
- Pretrained **AlexNet** weights are used for initial layers.
- Custom layers are trained for classification.
- Hyperparameters are set during training.

### 🧪 Image Classification
- Spectral features like texture and density are used.
- Classification is handled by the CNN and trained layers.

### 🌐 Web App using Flask
- A minimal Flask app interfaces with the trained model.
- Frontend: HTML, CSS, JS for input and output.
- Backend: Python (Flask) for prediction.

### 🚀 Model Deployment
- Model is deployed on **Heroku**.
- Uses **Git LFS** to upload large models (>100MB).
- Add the Git LFS buildpack on Heroku to avoid deployment issues.

---

## 🧾 Assumptions

- The model is trained on **specific crops only**.
- It can only detect diseases it was trained for.

**Supported Crops (15 total):**  
*(Insert image or list crops here if image not visible)*

**Detected Diseases (38 classes):**  
*(Insert image or list diseases here if image not visible)*

---

## 🛠️ Technologies & Tools

- **Keras** – Deep learning API for model creation
- **TensorFlow** – Backend for Keras
- **Jupyter Notebook / Google Colab** – Training platform with GPU support
- **Python** – Core programming language
- **Flask** – Web framework for deployment
- **HTML, CSS, JavaScript** – Frontend development
- **GitHub** – Source code versioning
- **Git LFS** – Handling large files like models
- **Heroku** – App hosting and deployment platform

---

## ✅ Advantages

- One-touch crop disease detection
- Saves time and manual effort
- Supports precision agriculture by improving crop yield quality

---

## 🔮 Future Scope

- Add recommendations and treatment options
- Expand model to support more crops and diseases
- Add multilingual support for wider farmer accessibility

---

## 🧾 Conclusion

This project enables farmers and agriculture professionals to detect crop diseases quickly and efficiently. It uses a deep learning-based approach that can be easily accessed via a simple web app. The system can be extended to include treatment recommendations and more crop classes.

---

## 📂 Dataset

- [Google Dataset Search – Plant Diseases](https://datasetsearch.research.google.com/search?query=plant-diseasesdataset&docid=ouHePAWoVIMq2IHEAAAAAA%3D%3D)

