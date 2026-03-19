import sounddevice as sd
from scipy.io.wavfile import write
from predict import predict_emotion

DURATION = 3
RATE = 22050

def record():
    print("Recording...")
    audio = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=1)
    sd.wait()
    write("temp.wav", RATE, audio)
    return "temp.wav"

file = record()
print("Emotion:", predict_emotion(file))