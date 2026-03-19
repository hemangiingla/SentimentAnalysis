import librosa
import numpy as np
import pickle
from tensorflow.keras.models import load_model

model = load_model("models/audio_model.h5")
le = pickle.load(open("models/label_encoder.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.hstack([mfcc, delta, delta2])

def predict_emotion(file):
    audio, sr = librosa.load(file, res_type='kaiser_fast')

    features = extract_features(audio, sr)
    features = scaler.transform([features])

    features = np.expand_dims(features, axis=2)

    pred = model.predict(features)
    return le.inverse_transform([np.argmax(pred)])[0]