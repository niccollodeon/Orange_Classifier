import { useState, useCallback, useRef } from "react";
import "./App.css";

const API_URL = "https://orange-classifier-1.onrender.com";

function ConfidenceBar({ value, isOrange }) {
  return (
    <div className="confidence-track">
      <div
        className={`confidence-fill ${isOrange ? "fill-orange" : "fill-gray"}`}
        style={{ width: `${value}%` }}
      />
    </div>
  );
}

function ResultCard({ result, previewUrl }) {
  const { label, is_orange, confidence } = result;
  return (
    <div className={`result-card ${is_orange ? "result-positive" : "result-negative"}`}>
      <div className="result-image-wrap">
        <img src={previewUrl} alt="Uploaded fruit" className="result-image" />
        <div className={`result-badge ${is_orange ? "badge-orange" : "badge-gray"}`}>
          {is_orange ? "🍊" : "✗"}
        </div>
      </div>
      <div className="result-body">
        <p className="result-label">{label}</p>
        <p className="result-sublabel">
          {is_orange
            ? "Our model detected an orange in this image."
            : "No orange detected in this image."}
        </p>
        <div className="confidence-row">
          <span className="confidence-text">Confidence</span>
          <span className="confidence-value">{confidence.toFixed(1)}%</span>
        </div>
        <ConfidenceBar value={confidence} isOrange={is_orange} />
      </div>
    </div>
  );
}

function UploadZone({ onFile, isDragging, setIsDragging }) {
  const inputRef = useRef(null);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) onFile(file);
    },
    [onFile, setIsDragging]
  );

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) onFile(file);
  };

  return (
    <div
      className={`upload-zone ${isDragging ? "dragging" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={handleChange}
      />
      <div className="upload-icon">
        <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
          <circle cx="20" cy="20" r="19" stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 3" />
          <path d="M20 26V14M14 20l6-6 6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
      <p className="upload-primary">Drop an image here</p>
      <p className="upload-secondary">or click to browse — JPG, PNG, WEBP</p>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = useCallback((f) => {
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("image", file);
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to connect to the backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      {/* Decorative background blobs */}
      <div className="blob blob-1" />
      <div className="blob blob-2" />
      <div className="blob blob-3" />

      <main className="main">
        {/* Header */}
        <header className="header">
          <div className="logo-wrap">
            <span className="logo-emoji">🍊</span>
          </div>
          <h1 className="title">OrangeID</h1>
          <p className="subtitle">
            CNN-powered citrus classifier — upload any image and let the model decide.
          </p>
        </header>

        {/* Card */}
        <div className="card">
          {!file ? (
            <UploadZone
              onFile={handleFile}
              isDragging={isDragging}
              setIsDragging={setIsDragging}
            />
          ) : (
            <div className="preview-wrap">
              <img src={previewUrl} alt="Preview" className="preview-img" />
              <button className="btn-ghost small" onClick={handleReset}>
                ✕ Remove
              </button>
            </div>
          )}

          {file && !result && (
            <button
              className={`btn-primary ${loading ? "loading" : ""}`}
              onClick={handlePredict}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Analysing…
                </>
              ) : (
                "Classify Image"
              )}
            </button>
          )}

          {error && (
            <div className="error-box">
              <strong>Connection error</strong>
              <p>{error}</p>
              <p className="error-hint">
                Make sure the Flask backend is running on <code>localhost:5000</code>.
              </p>
            </div>
          )}

          {result && (
            <>
              <ResultCard result={result} previewUrl={previewUrl} />
              <button className="btn-ghost" onClick={handleReset}>
                Try another image →
              </button>
            </>
          )}
        </div>

        {/* How it works */}
        <section className="how-it-works">
          <h2 className="section-title">How it works</h2>
          <div className="steps">
            {[
              { n: "01", label: "Upload", desc: "Select any fruit or food photo from your device." },
              { n: "02", label: "Inference", desc: "Your image is sent to a CNN trained on citrus imagery." },
              { n: "03", label: "Result", desc: "The model returns a classification with a confidence score." },
            ].map((s) => (
              <div className="step" key={s.n}>
                <span className="step-num">{s.n}</span>
                <strong className="step-label">{s.label}</strong>
                <p className="step-desc">{s.desc}</p>
              </div>
            ))}
          </div>
        </section>
      </main>

      <footer className="footer">
        <p>Powered by TensorFlow · Flask · React</p>
      </footer>
    </div>
  );
}