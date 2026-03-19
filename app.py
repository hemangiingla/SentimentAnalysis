from flask import Flask, render_template, request, jsonify
from predict import predict_emotion
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# 🎯 File Upload Prediction
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"})

    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    result = predict_emotion(path)
    return jsonify({"emotion": result})

# 🎤 Mic Recording Prediction
@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    audio = request.files.get("audio")

    if not audio:
        return jsonify({"error": "No audio received"})

    path = os.path.join(app.config['UPLOAD_FOLDER'], "recorded.wav")
    audio.save(path)

    result = predict_emotion(path)
    return jsonify({"emotion": result})

if __name__ == "__main__":
    app.run(debug=True)