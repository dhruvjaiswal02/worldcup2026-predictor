# 🏆 FIFA World Cup 2026 Predictor

An ML-powered tournament predictor built with Streamlit, using Elo ratings, squad strength intelligence, and Monte Carlo simulation.

**Project Overview**
This system simulates the entire 2026 FIFA World Cup thousands of times and produces probability estimates for every team at every stage of the tournament. Rather than giving a single prediction, it runs the tournament 1,000 to 10,000 times and counts how often each team wins — giving a realistic probability distribution that accounts for the inherent uncertainty of football.

**Dataset**
51,491 international matches from 1872 to 2026
Fields: Date, Home Team, Away Team, Scores, Tournament Type, Neutral Ground

**How It Works**
**1. Elo Rating System**
Every team starts at 1500. Ratings update after every match based on result and opponent strength. Stronger opponents beaten = bigger rating gain.
**2. K-Factor (Match Importance)**
World Cup matches update ratings 3x more than friendlies. K = 60 for World Cup, 40 for continental cups, 30 for competitive matches, 20 for friendlies.
**3. Time Decay**
Recent matches matter more. A match from 10 years ago carries 70% of the weight of a recent one, dropping to a minimum of 20% for very old matches.
**4. Goal Difference Multiplier**
Winning 5-0 matters more than 1-0. A logarithmic multiplier rewards dominant wins without letting heavy scorelines distort ratings.
**5. Home vs Neutral Ground**
During training, home teams receive a +100 Elo boost. All World Cup matches are on neutral ground so no boost is applied during simulation.
**6. Squad Player Ratings**
All 48 squads have hand-rated players (0–100 scale). The top 5 players per team are averaged into a squad strength score, mapped to an Elo boost or penalty:
Squad Boost = (Squad Strength − 75) × 4
**7. CONCACAF Correction**
Mexico, USA, and Canada have inflated Elos from playing weak regional opposition. Manual corrections applied: Mexico −120, USA −80, Canada −60.
**8. Match Prediction**
Win probability uses the standard Elo formula. Draw rate is calibrated to real World Cup data (27% for evenly matched teams, dropping as the gap widens, minimum 10%).
**9. Official 2026 Bracket**
Follows the real FIFA structure — 12 groups, Round of 32 with runner-up vs runner-up, winner vs crossing runner-up, and winner vs best third-place pairings. Fixed bracket paths ensure same-group teams can't meet before the Semi-Finals.
10. Monte Carlo Simulation
The full tournament is simulated up to 10,000 times. Championship probability = times a team won ÷ total simulations.
