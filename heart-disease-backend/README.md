# Heart Disease Prediction — Backend API

> Node.js · Express · MongoDB · Python ML Bridge  
> RESTful backend for the Heart Disease Prediction web application built on the UCI Cleveland dataset.

---

## Project Structure

```
heart-disease-backend/
├── src/
│   ├── config/
│   │   ├── db.js           → MongoDB connection (Mongoose)
│   │   ├── cors.js         → CORS allowed-origins config
│   │   └── jwt.js          → JWT secret & expiry config
│   ├── controllers/
│   │   ├── auth.controller.js        → Register, login, refresh, logout
│   │   ├── prediction.controller.js  → POST /predict, GET /predict/:id, stats
│   │   ├── history.controller.js     → User prediction history CRUD
│   │   ├── resource.controller.js    → Project resources & team info
│   │   └── health.controller.js      → Server + DB health check
│   ├── middleware/
│   │   ├── auth.js          → JWT protect, optionalAuth, restrictTo
│   │   ├── errorHandler.js  → AppError class, global error handler, 404
│   │   └── validate.js      → express-validator rules + runner
│   ├── models/
│   │   ├── User.js          → User schema (bcrypt, tokens)
│   │   └── Prediction.js    → Prediction schema (13 UCI features + result)
│   ├── routes/
│   │   ├── auth.routes.js        → /api/v1/auth/*
│   │   ├── prediction.routes.js  → /api/v1/predict/*
│   │   ├── history.routes.js     → /api/v1/history/*
│   │   ├── resource.routes.js    → /api/v1/resources
│   │   └── health.routes.js      → /api/v1/health
│   ├── services/
│   │   ├── prediction.service.js → Python bridge + JS fallback
│   │   └── auth.service.js       → JWT verify helper
│   ├── utils/
│   │   ├── logger.js    → Winston logger (console + file)
│   │   ├── response.js  → sendSuccess / sendError helpers
│   │   └── features.js  → UCI feature formatting & sanitization
│   └── app.js           → Express app (middleware, routes)
├── ml/
│   ├── predict_bridge.py  → Reads JSON from stdin, outputs prediction JSON
│   ├── train.py           → Trains ensemble model, saves model.pkl + scaler.pkl
│   └── requirements.txt   → Python ML dependencies
├── logs/                  → Auto-created at runtime
├── .env.example           → Environment variable template
├── .gitignore
├── package.json
├── server.js              → Entry point
└── README.md
```

---

## Quick Start

### 1. Prerequisites

| Tool | Version |
|------|---------|
| Node.js | ≥ 18 |
| MongoDB | ≥ 6 (local or Atlas) |
| Python | ≥ 3.9 (optional — for real ML model) |

### 2. Install & Configure

```bash
# Clone / enter project
cd heart-disease-backend

# Install Node dependencies
npm install

# Copy and edit environment file
cp .env.example .env
# → Edit MONGODB_URI and JWT_SECRET at minimum
```

### 3. (Optional) Train the ML Model

```bash
# Install Python dependencies
pip install -r ml/requirements.txt

# Train and save model.pkl + scaler.pkl
npm run train-model
# OR: python3 ml/train.py
# OR: python3 ml/train.py path/to/your/cleveland.csv
```

> **Without training:** The server still works using a built-in JavaScript logistic regression fallback with UCI Cleveland coefficients. The `modelUsed` field in responses will say `logistic_regression_fallback`.

### 4. Start the Server

```bash
# Development (auto-reload with nodemon)
npm run dev

# Production
npm start
```

Server starts at: `http://localhost:5001`

---

## 🌐 API Reference

### Base URL
```
http://localhost:5001/api/v1
```

### Authentication
All protected endpoints require:
```
Authorization: Bearer <accessToken>
```

---

###  Auth Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/auth/register` | No | Register new user |
| POST | `/auth/login` | No | Login |
| POST | `/auth/refresh` | No | Refresh access token |
| POST | `/auth/logout`  | Logout |
| GET | `/auth/me` | Get current user profile |
| PUT | `/auth/password` | Update password |

#### Register
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "name": "Ruha Kabbani",
  "email": "ruha@example.com",
  "password": "securepass123"
}
```

Response `201`:
```json
{
  "success": true,
  "accessToken": "eyJ...",
  "refreshToken": "eyJ...",
  "user": { "id": "...", "name": "Ruha Kabbani", "email": "ruha@example.com", "role": "user" }
}
```

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "ruha@example.com",
  "password": "securepass123"
}
```

---

### Prediction Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/predict` | Optional | Run heart disease prediction |
| GET | `/predict/stats` | Admin only | Global prediction statistics |
| GET | `/predict/:id` | Optional | Get a prediction record by ID |

#### Run Prediction
```http
POST /api/v1/predict
Content-Type: application/json

{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

Response `200`:
```json
{
  "success": true,
  "predictionId": "6654abc...",
  "result": {
    "prediction": 1,
    "probability": 0.7823,
    "probabilityPercent": "78.23",
    "riskLevel": "high",
    "modelUsed": "VotingClassifier",
    "featureImportance": {
      "cp": 0.231,
      "thalach": 0.178,
      "ca": 0.156
    }
  },
  "processingTimeMs": 142
}
```

#### UCI Cleveland Feature Guide

| Feature | Type | Range / Values | Description |
|---------|------|----------------|-------------|
| `age` | Number | 1–120 | Age in years |
| `sex` | 0/1 | 0=Female, 1=Male | Biological sex |
| `cp` | 0–3 | 0=Typical angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic | Chest pain type |
| `trestbps` | Number | 50–300 mm Hg | Resting blood pressure |
| `chol` | Number | 50–700 mg/dl | Serum cholesterol |
| `fbs` | 0/1 | 0=No, 1=Yes | Fasting blood sugar > 120 mg/dl |
| `restecg` | 0–2 | 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy | Resting ECG |
| `thalach` | Number | 50–250 bpm | Maximum heart rate achieved |
| `exang` | 0/1 | 0=No, 1=Yes | Exercise-induced angina |
| `oldpeak` | Number | 0–10 | ST depression induced by exercise |
| `slope` | 0–2 | 0=Upsloping, 1=Flat, 2=Downsloping | Slope of peak exercise ST segment |
| `ca` | 0–3 | Integer | Major vessels colored by fluoroscopy |
| `thal` | 1–3 | 1=Normal, 2=Fixed defect, 3=Reversible defect | Thalassemia |

---

### History Endpoints (Requires Auth)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/history` | List user's predictions (paginated) |
| GET | `/history/summary` | Summary stats (totals, risk breakdown) |
| GET | `/history/:id` | Single prediction record |
| DELETE | `/history/:id` | Delete one record |
| DELETE | `/history` | Clear all history |

Query params for `GET /history`:
- `page` (default: 1)
- `limit` (default: 10, max: 50)
- `riskLevel` — filter: `low`, `moderate`, `high`
- `prediction` — filter: `0` (no disease) or `1` (disease)

---

### Resource Endpoint

```http
GET /api/v1/resources
```
Returns project links, team info, technology list, and feature descriptions.

---

### Health Check

```http
GET /api/v1/health
```
Returns server status, DB connection state, uptime, Node version.

---

##  ML Bridge Architecture

```
React Frontend
     │
     ▼ POST /api/v1/predict  (13 features as JSON)
Express Route
     │
     ▼
PredictionService.runPrediction(features)
     │
     ├── spawn("python3", ["ml/predict_bridge.py"])
     │       │
     │       ├── Load model.pkl + scaler.pkl
     │       ├── Scale features (StandardScaler)
     │       └── Return { prediction, probability, model_used }
     │
     └── [Fallback if Python unavailable]
             └── JS Logistic Regression (UCI coefficients)
     │
     ▼
Save Prediction to MongoDB
     │
     ▼
Return JSON response to Frontend
```

---

## Security Features

- **Helmet** — HTTP security headers
- **CORS** — Allowlist-based origin control
- **Rate Limiting** — 200 req/15min globally; 10 req/min for predictions
- **JWT** — Short-lived access tokens (1h) + refresh tokens (7d)
- **bcryptjs** — Password hashing with salt rounds=12
- **Input Validation** — All 13 UCI features validated with express-validator
- **Error Sanitization** — No stack traces in production responses

---

##  ML Pipeline Summary

The Python model trained by `ml/train.py` replicates the team's notebook pipeline:

1. **KNN Imputer** — fills missing values (ca, thal have some in original dataset)
2. **IQR Clipping** — clips outliers for continuous features
3. **SMOTE** — oversamples minority class on training set only
4. **StandardScaler** — normalizes features
5. **Ensemble (VotingClassifier):**
   - Logistic Regression (L2)
   - Random Forest (200 trees)
   - SVM (RBF kernel)
   - XGBoost (if available)

---

##  Team

| Name | 
|------|
| Sidra Ashram 
| Layan Junaid |
| Ruha Kabbani | 

**Supervisor:** Shahaboddin Daneshvar

---

##  License

MIT — For educational purposes. Not a medical diagnostic tool.
