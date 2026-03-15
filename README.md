# 🍊 OrangeID — CNN Citrus Classifier

A full-stack web application that uses a fine-tuned EfficientNetB0 model to classify whether an image contains an orange or not. Upload any image and get an instant prediction with a confidence score.

**Live demo:** [your-app.vercel.app](https://orange-classifier-fbyd.vercel.app/) · **API:** [your-api.railway.app](https://orangeclassifier-production.up.railway.app)

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite |
| Backend | Python, Flask |
| ML Model | TensorFlow, EfficientNetB0 |
| Containerization | Docker |
| Frontend hosting | Vercel |
| Backend hosting | Railway |

---

## Project structure

```
Orange_Classifier/
├── .gitignore
├── vercel.json               ← Vercel build config
├── README.md
│
├── backend/
│   ├── app.py                ← Flask API
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .dockerignore
│   └── 1.keras               ← Trained EfficientNetB0 model
│
└── frontend/
    ├── src/
    │   ├── App.jsx           ← Main React component
    │   ├── App.css           ← Styles
    │   └── main.jsx          ← Entry point
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## Model details

- **Architecture:** EfficientNetB0 (pretrained on ImageNet, fine-tuned)
- **Input size:** 110 × 110 × 3
- **Output:** 2-class softmax — `Not Orange` (index 0) / `Orange` (index 1)
- **Preprocessing:** `tensorflow.keras.applications.efficientnet.preprocess_input`
- **Training:** Google Colab, saved in `.keras` format

### Model architecture

```
EfficientNetB0 (frozen, pooling=avg)
→ Flatten
→ Dense(64, relu) + L2(0.01)
→ BatchNormalization
→ Dropout(0.5)
→ Dense(32, relu)
→ BatchNormalization
→ Dropout(0.2)
→ Dense(2, softmax)
```

---

## API reference

Base URL: `https://orangeclassifier-production.up.railway.app`

### `GET /health`

Returns the server and model status.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "/app/1.keras"
}
```

### `POST /predict`

Accepts a `multipart/form-data` request with an `image` field.

**Request**
```bash
curl -X POST https://orangeclassifier-production.up.railway.app/predict \
  -F "image=@orange.jpg"
```

**Response**
```json
{
  "label": "Orange",
  "is_orange": true,
  "confidence": 94.23,
  "raw_score": 0.9423
}
```

| Field | Type | Description |
|---|---|---|
| `label` | string | `"Orange"` or `"Not Orange"` |
| `is_orange` | boolean | `true` if orange detected |
| `confidence` | number | Confidence percentage (0–100) |
| `raw_score` | number | Raw model output (0–1) |

---

## Running locally

### Prerequisites

- Python 3.11+
- Node.js 18+
- Your trained `.keras` model file in `backend/`

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

API runs on **http://localhost:5000**

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App runs on **http://localhost:3000**

> Make sure `API_URL` in `frontend/src/App.jsx` points to `http://localhost:5000` for local development.

---

## Deployment

This project uses a single GitHub repository with two separate deployments.

### Backend → Railway

1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Select the repo and set **Root Directory** to `backend`
3. Railway auto-detects the `Dockerfile` and builds the container
4. Copy the generated URL (e.g. `https://your-api.railway.app`)

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) → New Project → Import from GitHub
2. Set **Root Directory** to `frontend`
3. Update `API_URL` in `App.jsx` to your Railway URL
4. Vercel auto-detects Vite and deploys

```
GitHub repo
├── frontend/  →  Vercel  →  https://orange-classifier-fbyd.vercel.app/
└── backend/   →  Railway →  https://orangeclassifier-production.up.railway.app
```

---

## Environment notes

- The backend uses `tensorflow-cpu` — no GPU required
- EfficientNetB0 base layers are frozen during inference
- CORS is enabled for all origins via `flask-cors`
- The model is loaded once at startup and reused across requests

---