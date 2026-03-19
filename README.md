# 🎤 Emotion AI Analyzer

### Speech Emotion Recognition using Deep Learning (CNN + LSTM)

---

## 📌 Overview

Emotion AI Analyzer is a **Speech Emotion Recognition (SER) system** that detects human emotions from audio signals using **Deep Learning techniques**.

The system supports:

* 🎤 Real-time voice recording (browser-based)
* 📂 Audio file upload
* 🤖 Emotion prediction using CNN + LSTM model
* 📊 Visualization of accuracy and confusion matrix

---

## 🚀 Features

* 🎤 **Live Microphone Recording (Web-based)**
* 📂 **Upload Audio Files (.wav)**
* 🧠 **Deep Learning Model (CNN + LSTM)**
* 📈 **Accuracy Graph & Confusion Matrix**
* 🎨 **Modern UI (Glassmorphism Design)**
* ⚡ **Real-time Emotion Prediction API**

---

## 🧠 Technologies Used

* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Python (Flask)
* **Machine Learning:** TensorFlow / Keras
* **Audio Processing:** Librosa
* **Visualization:** Matplotlib, Seaborn

---

## 📂 Project Structure

```
emotion_ai_final/
│
├── app.py
├── train_model.py
├── predict.py
├── mic_predict.py
├── requirements.txt
│
├── dataset/
├── models/
├── uploads/
├── results/
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/emotion-ai-analyzer.git
cd emotion-ai-analyzer
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the model

```bash
python train_model.py
```

### 4️⃣ Run the web app

```bash
python app.py
```

### 5️⃣ Open in browser

```
http://127.0.0.1:5000/
```

---

## 🎤 How to Use

### 📂 Upload Audio

1. Click "Upload Audio"
2. Select `.wav` file
3. Click **Analyze**
4. View detected emotion

### 🎙 Record Audio

1. Click **Start Recording**
2. Speak for a few seconds
3. Click **Stop Recording**
4. Emotion will be displayed

---

## 📊 Model Details

* **Architecture:** CNN + LSTM Hybrid
* **Input Features:**

  * MFCC
  * Delta & Delta-Delta Features
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Accuracy:** ~85–92% (depending on dataset)

---

## 📈 Output

* 📊 Accuracy Graph (`results/accuracy.png`)
* 📉 Confusion Matrix (`results/confusion_matrix.png`)
* 🎯 Predicted Emotion Output

---

## 🎯 Applications

* 🧠 Mental health monitoring
* 📞 Call center analytics
* 🎮 Gaming emotion detection
* 🤖 Human-computer interaction

---

## ⚠️ Limitations

* Requires clean audio input
* Accuracy depends on dataset size
* Limited real-world noise handling

---

## 🔮 Future Enhancements

* 🎤 Real-time continuous emotion tracking
* 🧠 Integration with advanced models (Wav2Vec / Transformers)
* 📱 Mobile app deployment
* ☁️ Cloud deployment (AWS / GCP)

