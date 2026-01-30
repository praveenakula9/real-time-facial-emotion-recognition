# 🎭 Real-Time Facial Emotion Recognition (Local)

This project implements a **real-time facial emotion recognition system** using **Deep Learning**.  
It captures live video from a webcam, detects faces, and predicts human emotions in real time.

> ⚠️ **Note:** This project is intended for **local execution only** and is **not suitable for Streamlit Cloud or browser-based deployment**.

---

## ✨ Features
- Real-time face detection using **MediaPipe (Tasks API)**
- Emotion classification using a **CNN model (TensorFlow/Keras)**
- Webcam-based live inference
- Emotion label + confidence display
- FPS counter for performance monitoring
- Automatic MediaPipe model download
- Large model files managed using **Git LFS**

---

## 🧠 Emotions Detected
- Angry  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

---

## 🛠 Tech Stack
- **Python 3.11**
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy

---

## 📁 Project Structure

```text
.
├── app.py                # Main real-time emotion detection script
├── best_model.keras      # Trained CNN model (Git LFS)
├── requirements.txt      # Python dependencies
├── .gitattributes        # Git LFS configuration
└── README.md
```
---


## 🧠 How It Works
1. Webcam captures video frames
2. MediaPipe detects faces
3. Face region is resized and normalized
4. CNN model predicts emotion
5. Emotion label and confidence are shown on screen

---

## 🚀 Quick Start (Local Execution)

Follow these steps to run the project on your local machine:

```bash
git clone https://github.com/praveenakula9/real-time-facial-emotion-recognition.git
cd real-time-facial-emotion-recognition
python -m venv mp_env
mp_env\Scripts\activate
pip install -r requirements.txt
python app.py
```
---

## 📸 Screenshots

### Real-Time Emotion Detection

Below is a sample output of the application running locally:

![Real-Time Emotion Detection](screenshots/emotion_detection.png)

**Visible elements:**
- Face bounding box detected by MediaPipe  
- Predicted emotion label (e.g., HAPPY, SAD)  
- Confidence percentage  
- FPS counter  


