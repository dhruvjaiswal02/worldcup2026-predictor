# 🏆 FIFA World Cup 2026 Predictor

An ML-powered tournament predictor built with Streamlit, using Elo ratings, squad strength intelligence, and Monte Carlo simulation.

## Features

- **Championship Odds** — Monte Carlo probabilities for all 48 teams
- **Stage-by-Stage Breakdown** — Heatmap showing how far each team goes
- **Group Stage** — Per-group qualification probabilities
- **Team Ratings** — Squad-adjusted Elo ratings with player quality boost
- **Upset Radar** — Most likely shock results ranked by probability
- **Head-to-Head** — Match probability between any two teams (sidebar)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (Free)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/wc2026-predictor.git
   git push -u origin main
   ```

2. **Deploy**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **"New app"**
   - Select your repo → branch: `main` → file: `app.py`
   - Click **Deploy** — you'll get a public URL in ~2 minutes

3. **Your public URL will be:**
   ```
   https://YOUR_USERNAME-wc2026-predictor-app-XXXXX.streamlit.app
   ```

## Tech Stack

- **Elo Rating System** — trained on historical international match data
- **Squad Strength** — top-5 player ratings per team mapped to Elo boost/penalty
- **Monte Carlo Simulation** — 100 to 10,000 tournament runs
- **Plotly** — interactive charts
- **Streamlit** — web app framework

## Note

This app is self-contained — no external CSV required. Base Elo ratings are pre-calibrated from historical data. To improve accuracy, connect to the live Elo pipeline from the full Jupyter notebook (`Worldcupresultspredictor_WithSquads.ipynb`).
