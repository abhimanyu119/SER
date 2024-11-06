from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

app = Flask(__name__)
cors = CORS(app)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


class EmotionDatasetConfig:
    EMOTION_MAPPING = {
        "angry": 0,
        "anger": 0,
        "ANG": 0,
        "03": 0,
        "ANGRY": 0,
        "disgust": 1,
        "disgust": 1,
        "DIS": 1,
        "07": 1,
        "DISGUST": 1,
        "fear": 2,
        "fearful": 2,
        "FEA": 2,
        "06": 2,
        "FEAR": 2,
        "happy": 3,
        "happiness": 3,
        "HAP": 3,
        "04": 3,
        "HAPPY": 3,
        "neutral": 4,
        "neutral": 4,
        "NEU": 4,
        "01": 4,
        "NEUTRAL": 4,
        "sad": 5,
        "sadness": 5,
        "SAD": 5,
        "05": 5,
        "SAD": 5,
        "surprise": 6,
        "surprised": 6,
        "SUR": 6,
        "08": 6,
        "SURPRISE": 6,
    }

    DATASET_CONFIGS = {
        "CREMA-D": {
            "file_pattern": "*.wav",
            "emotion_parser": lambda x: x.stem.split("_")[2],
        },
        "RAVDESS": {
            "file_pattern": "*.wav",
            "emotion_parser": lambda x: x.stem.split("-")[2],
        },
        "SAVEE": {
            "file_pattern": "*.wav",
            "emotion_parser": lambda x: x.stem[:3],
        },
        "TESS": {
            "file_pattern": "*.wav",
            "emotion_parser": lambda x: x.stem.split("_")[-1],
        },
        "ESD": {
            "file_pattern": "*.wav",
            "emotion_parser": lambda x: x.stem.split("_")[2],
        },
    }


class AudioDataset(Dataset):
    def __init__(self, data_dir, dataset_name, sr=16000, duration=3):
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.sr = sr
        self.duration = duration
        self.config = EmotionDatasetConfig.DATASET_CONFIGS[dataset_name]
        self.files = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        for file in self.data_dir.rglob(self.config["file_pattern"]):
            try:
                emotion = self.config["emotion_parser"](file)
                emotion_idx = EmotionDatasetConfig.EMOTION_MAPPING.get(emotion)

                if emotion_idx is not None:
                    self.files.append(file)
                    self.labels.append(emotion_idx)
            except:
                continue

        def process_audio(self, audio_path):  # Changed from _process_audio
            try:
                signal, sr = librosa.load(audio_path, sr=self.sr)

                target_length = self.sr * self.duration
                if len(signal) > target_length:
                    signal = signal[:target_length]
                else:
                    signal = np.pad(signal, (0, max(0, target_length - len(signal))))

                mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
                mfccs_scaled = np.mean(mfccs.T, axis=0)
                delta = librosa.feature.delta(mfccs)
                delta2 = librosa.feature.delta(mfccs, order=2)

                features = np.concatenate([mfccs, delta, delta2])
                features = torch.FloatTensor(features).unsqueeze(0)

                return features
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                return None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]

        features = self.process_audio(audio_path)
        if features is None:
            features = torch.zeros((1, 120, 94))

        return features, label


class EmotionRecognitionModel(nn.Module):
    def __init__(self, n_features=120, n_classes=7):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.fc_net = nn.Sequential(
            nn.Linear(256 * (n_features // 8) * (94 // 8), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        x = self.fc_net(x)
        return x


def train_model(dataset_paths, batch_size=32, epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionRecognitionModel().to(device)

    datasets = []
    for dataset_path, dataset_name in dataset_paths.items():
        dataset = AudioDataset(dataset_path, dataset_name)
        datasets.append(dataset)

    combined_dataset = ConcatDataset(datasets)
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {train_loss/10:.4f}, Accuracy: {100.*train_correct/train_total:.2f}%"
                )
                train_loss = 0

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                "best_model.pth",
            )

        print(
            f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {100.*val_correct/val_total:.2f}%"
        )


@app.route("/train", methods=['POST', 'OPTIONS'])
@cross_origin()
def train_endpoint():
    try:
        dataset_paths = request.get_json()
        train_model(dataset_paths)
        return jsonify({"message": "Training completed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=['POST', 'OPTIONS'])
@cross_origin()
def predict_emotion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionRecognitionModel().to(device)

    if os.path.exists("best_model.pth"):
        checkpoint = torch.load("best_model.pth")
        model.load_state_dict(checkpoint["model_state_dict"])

    file = request.files["file"]
    with open("temp_audio.wav", "wb") as f:
        f.write(file.read())

    dataset = AudioDataset("", "CREMA-D")
    print("Attempting to process audio file")
    features = dataset.process_audio("temp_audio.wav")
    features = features.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(features)
        prediction = torch.softmax(output, dim=1)
        emotion_idx = prediction.argmax().item()
        confidence = prediction[0][emotion_idx].item()

    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    return jsonify({"emotion": emotions[emotion_idx], "confidence": confidence})


if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)
