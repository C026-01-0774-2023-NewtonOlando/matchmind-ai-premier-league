# ⚽ MatchMind AI — Explainable Match Intelligence & Multi-Agent Hub
**Microsoft Premier League Hackathon Submission**

## Overview
MatchMind AI transforms raw football video and synthetic tracking telemetry into explainable match intelligence, individual player scouting dossiers, and personalized multi-language narratives.

### 5-Stage Hackathon Pipeline
1. **Ingest (`event_detector.py`):** Custom YOLO (`best.pt`) + BoT-SORT (`tracker.yaml`) auto-eventing pipeline with camera motion compensation and deterministic K-Means team clustering.
2. **Interpret (`player_report_engine.py`):** Computes calibrated Pass Distance (m), Ball Speed (km/h), Corridor Congestion, Pass Difficulty Rating (0–100), Expected Pass Completion (xP%), Player Sprint Speed, and Dribble/Ball-Carry Distance.
3. **Explain:** Second-by-second **Control vs. Chaos Index (-100 to +100)** and **Tactical Pressure Index (0–100)** explaining transitions from turnover chaos to structured buildup control.
4. **Render (`app.py`):** Interactive Streamlit hub featuring a 2D Synthetic Digital Twin Pitch, synchronized H.264 CV video feed, and lower-third broadcast graphic overlays.
5. **Personalize:** Microsoft AI Foundry multi-agent orchestration adapting insights across **Analyst Mode**, **Fan Mode**, and **Broadcast Mode** with **Player-Focused Spotlight Mode** in **English, Spanish, French, and Swahili**.

## Quickstart
```bash
pip install -r requirements.txt
python player_report_engine.py
streamlit run app.py