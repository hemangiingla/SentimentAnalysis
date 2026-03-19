let mediaRecorder;
let audioChunks = [];

// 🎤 Start Recording
async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.start();
    audioChunks = [];

    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };

    document.getElementById("result").innerText = "Recording...";
}

// 🛑 Stop Recording
function stopRecording() {
    mediaRecorder.stop();

    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });

        const formData = new FormData();
        formData.append("audio", audioBlob, "recorded.wav");

        const response = await fetch("/predict_audio", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        document.getElementById("result").innerText = "Emotion: " + data.emotion;
    };
}

// 📂 Upload File
async function uploadFile() {
    const fileInput = document.getElementById("fileInput");
    const formData = new FormData();

    formData.append("file", fileInput.files[0]);

    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();
    document.getElementById("result").innerText = "Emotion: " + data.emotion;
}