import { useState, useRef } from 'react';
import { Mic, Upload, Folder } from 'lucide-react';

const EmotionRecognition = () => {
  const [recording, setRecording] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [message, setMessage] = useState(null);
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];

      mediaRecorder.current.ondataavailable = (event) => {
        audioChunks.current.push(event.data);
      };

      mediaRecorder.current.onstop = async () => {
        const audioBlob = new Blob(audioChunks.current, { type: 'audio/wav' });
        await predictEmotion(audioBlob);
      };

      mediaRecorder.current.start();
      setRecording(true);
      setMessage(null);
    } catch (error) {
      setMessage({ type: 'error', text: 'Error accessing microphone: ' + error.message });
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && recording) {
      mediaRecorder.current.stop();
      setRecording(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      await predictEmotion(file);
    }
  };

  const handleFolderUpload = async (event) => {
    const files = event.target.files;
    if (files.length > 0) {
      setTraining(true);
      setMessage(null);
      try {
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
          formData.append('files', files[i]);
          setTrainingProgress((i / files.length) * 100);
        }

        const response = await fetch('http://localhost:8000/train', {
          method: 'POST',
          body: formData,
        });

        const result = await response.json();
        setMessage({ type: 'success', text: 'Training completed successfully!' });
      } catch (error) {
        setMessage({ type: 'error', text: 'Error during training: ' + error.message });
      }
      setTraining(false);
      setTrainingProgress(0);
    }
  };

  const predictEmotion = async (audioData) => {
    setLoading(true);
    setMessage(null);
    try {
      const formData = new FormData();
      formData.append('file', audioData);

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      setMessage({ type: 'error', text: 'Error predicting emotion: ' + error.message });
    }
    setLoading(false);
  };

  const getEmotionColor = (emotion) => {
    const colors = {
      angry: 'red',
      disgust: 'green',
      fear: 'purple',
      happy: 'yellow',
      neutral: 'gray',
      sad: 'blue',
      surprise: 'pink',
    };
    return colors[emotion] || 'gray';
  };

  return (
    <div className="card">
      <header className="card-header">
        <h2>Speech Emotion Recognition</h2>
      </header>
      <div className="card-content">
        <div className="button-group">
          <button onClick={recording ? stopRecording : startRecording} className={`button ${recording ? 'recording' : ''}`} disabled={training}>
            <Mic className="icon" />
            {recording ? 'Stop Recording' : 'Start Recording'}
          </button>
          <button onClick={() => document.getElementById('file-upload').click()} className="button" disabled={training}>
            <Upload className="icon" />
            Upload Audio
          </button>
          <button onClick={() => document.getElementById('folder-upload').click()} className="button" disabled={training}>
            <Folder className="icon" />
            Train Model
          </button>
        </div>

        <input id="file-upload" type="file" accept="audio/*" className="hidden" onChange={handleFileUpload} />
        <input id="folder-upload" type="file" webkitdirectory="true" multiple className="hidden" onChange={handleFolderUpload} />

        {training && (
          <div className="training-info">
            <p>Training in progress...</p>
            <progress value={trainingProgress} max="100"></progress>
          </div>
        )}

        {loading && <p>Processing audio...</p>}

        {message && (
          <div className={`alert ${message.type === 'error' ? 'alert-error' : 'alert-success'}`}>
            <p>{message.text}</p>
          </div>
        )}

        {prediction && (
          <div className="prediction">
            <span className={`emotion ${getEmotionColor(prediction.emotion)}`}>
              {prediction.emotion.toUpperCase()}
            </span>
            <p>Confidence: {(prediction.confidence * 100).toFixed(2)}%</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default EmotionRecognition;
