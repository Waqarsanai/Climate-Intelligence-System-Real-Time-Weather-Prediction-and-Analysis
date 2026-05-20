# Deployment Guide (Vercel + Render)

This project is best deployed as:
- **Frontend** on **Vercel** (static UI)
- **Backend** on **Render** (Flask API)

---

## 1) Deploy the Backend (Render)

1. Go to Render and create a **New Web Service** from this repo.
2. Use these settings (Render will also auto-read `render.yaml`):
   - **Name**: `climate-intelligence-api` (so the default URL works)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -w 2 -k gthread -t 120 -b 0.0.0.0:$PORT weather_app.api_server:create_app()`
   - **Health Check Path**: `/health`
3. Deploy. Your backend URL will be:
   - `https://climate-intelligence-api.onrender.com`

> If you choose a different service name, update `config.js` accordingly.

---

## 2) Deploy the Frontend (Vercel)

1. Import the same repo into Vercel.
2. Framework preset: **Other** (static).
3. Deploy. The root URL will load `app.html` via `vercel.json`.

---

## 3) Verify It Works

- Open your Vercel URL in the browser.
- The UI should load and call the Render API.
- Check the backend health:
  - `https://climate-intelligence-api.onrender.com/health`

---

## Notes

- Free Render services may “sleep” on inactivity; the first request can take 30–60 seconds.
- If you want to fully train models in production, add heavy ML deps (xgboost, lightgbm, catboost, tensorflow) to `requirements.txt`.
