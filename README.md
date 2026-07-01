# 🩺 AI-Based Multi-Cancer Detection and Localization System Using Deep Learning and Medical Image Analytics

## 📌 Overview

This project is an AI-powered web application that detects multiple types of cancer from medical images using Deep Learning and Explainable AI (XAI). The system currently supports Brain Tumor detection from MRI images and Lung Cancer detection from Chest X-ray images. It also provides Grad-CAM heatmaps to highlight the regions influencing the model's prediction.

---

## ✨ Features

* 🧠 Brain Tumor Detection (MRI)
* 🫁 Lung Cancer Detection (Chest X-ray)
* 📤 Upload medical images through a web interface
* 🤖 Automatic prediction using Deep Learning
* 🔥 Grad-CAM visualization for explainability
* 📊 Confidence score for predictions
* 🌐 Flask-based web application
* 📱 Responsive user interface

---

## 🏗 Project Architecture

```
Medical Image
      │
      ▼
Image Preprocessing
      │
      ▼
Deep Learning Model (ResNet18)
      │
      ▼
Cancer Classification
      │
      ▼
Grad-CAM Localization
      │
      ▼
Prediction + Confidence Score
      │
      ▼
Web Application
```

---

## 📂 Project Structure

```
multi-cancer-detection/

├── backend/
│   ├── app.py
│   ├── models/
│   │   ├── brain_model.pth
│   │   ├── brain_classes.json
│   │   ├── lung_model.pth
│   │   └── lung_classes.json
│   ├── uploads/
│   └── static/
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
├── notebooks/
│   ├── brain_training.ipynb
│   └── lung_training.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🧠 Deep Learning Models

### Brain Tumor Detection

* Model: ResNet18 (Transfer Learning)
* Dataset: Brain Tumor MRI Dataset (Kaggle)
* Classes:

  * Glioma
  * Meningioma
  * Pituitary
  * No Tumor
* Test Accuracy: **95.31%**

---

### Lung Cancer Detection

* Model: ResNet18 (Transfer Learning)
* Dataset: Lung Cancer Chest X-ray Dataset (Kaggle)
* Classes:

  * Cancer
  * Normal
* Test Accuracy: **89.42%**

---

## 📚 Datasets

### Brain MRI Dataset

* Source: Kaggle
* Images: ~7000+
* Type: MRI

### Lung Cancer Dataset

* Source: Kaggle
* Type: Chest X-ray

---

## 🛠 Technologies Used

### Programming Language

* Python

### Deep Learning

* PyTorch
* TorchVision

### Image Processing

* OpenCV
* Pillow
* NumPy

### Explainable AI

* Grad-CAM

### Backend

* Flask
* Flask-CORS

### Frontend

* HTML
* CSS
* JavaScript
* Bootstrap

### Development Tools

* VS Code
* Google Colab
* Git
* GitHub

---

## 📊 Model Evaluation

### Brain MRI

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 95.31% |

### Lung X-ray

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 89.42% |

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/your-username/multi-cancer-detection.git
```

Go to project directory

```bash
cd multi-cancer-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run backend

```bash
cd backend
python app.py
```

Run frontend

```bash
cd frontend
python -m http.server 5500
```

Open your browser

```
http://127.0.0.1:5500
```

---

## 📷 Application Workflow

1. Upload a medical image.
2. The system preprocesses the image.
3. The trained model predicts the cancer type.
4. Confidence score is displayed.
5. Grad-CAM heatmap highlights important regions.
6. Results are presented through the web interface.

---

## 🔮 Future Scope

* Breast Cancer Detection
* Skin Cancer Detection
* Automatic Medical Image Type Recognition
* U-Net-based Tumor Segmentation
* Cloud Deployment (AWS/GCP)
* User Authentication
* Database Integration
* Mobile Application

---

## 📖 Research Topic

**AI-Based Multi-Cancer Detection and Localization System Using Deep Learning and Medical Image Analytics**

---

## 👨‍💻 Author

**Aditya Satpathy**

Master of Computer Applications (MCA)

KIIT University

---

## 📜 License

This project is developed for academic and research purposes. It is **not intended for clinical diagnosis or medical decision-making**.
