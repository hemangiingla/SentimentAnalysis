import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM, BatchNormalization

# ✅ Auto create folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

DATASET_PATH = "dataset/"

# 🎯 Emotion mapping
emotions = {
    '01':'neutral','02':'calm','03':'happy','04':'sad',
    '05':'angry','06':'fear','07':'disgust','08':'surprise'
}

# 🔥 Feature extraction with delta
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.hstack([mfcc, delta, delta2])

# 🔥 Data augmentation
def augment(audio, sr):
    noise = np.random.randn(len(audio))
    audio_noise = audio + 0.005 * noise

    audio_pitch = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)

    return [audio, audio_noise, audio_pitch]

# 📊 Load dataset
X, y = [], []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            emotion = emotions.get(emotion_code)

            file_path = os.path.join(root, file)
            audio, sr = librosa.load(file_path, res_type='kaiser_fast')

            # 🔥 Apply augmentation
            for aug in augment(audio, sr):
                features = extract_features(aug, sr)
                X.append(features)
                y.append(emotion)

X = np.array(X)

# 🔥 Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

pickle.dump(scaler, open("models/scaler.pkl", "wb"))

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Reshape for CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 🔥 HIGH PERFORMANCE MODEL
model = Sequential()

model.add(Conv1D(128, 5, activation='relu', input_shape=(X.shape[1],1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))

model.add(LSTM(256))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(set(y)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 🚀 Train
history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save model
model.save("models/audio_model.h5")
pickle.dump(le, open("models/label_encoder.pkl", "wb"))

# 📈 Accuracy graph
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy Graph")
plt.savefig("results/accuracy.png")

# 📊 Confusion matrix
y_pred = np.argmax(model.predict(X_test), axis=1)

acc = accuracy_score(y_test, y_pred)
print("🔥 Final Accuracy:", acc)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")

print("✅ Training Complete!")