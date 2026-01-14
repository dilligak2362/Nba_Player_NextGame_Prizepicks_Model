🏀 NBA Player Prop Prediction & Edge Detection Engine

A full-stack machine learning pipeline for predicting NBA player props, detecting sportsbook edges, calibrating model bias, and tracking long-term betting performance.

This system ingests historical NBA data, trains stat-level regression models, generates daily projections, scrapes live sportsbook prop lines, identifies value edges, calibrates predictions using real outcomes, and tracks profitability across prop types, directions, and edge buckets.

Built for real-world sports trading workflows.

⸻

🚀 Features

📊 Predictive Modeling

Individual machine learning models for:
	•	Points (PTS)
	•	Rebounds (REB)
	•	Assists (AST)
	•	Steals (STL)
	•	Blocks (BLK)
	•	Turnovers (TO)
	•	Minutes projection

Combo projections:
	•	PR (Points + Rebounds)
	•	PA (Points + Assists)
	•	RA (Rebounds + Assists)
	•	PRA (Points + Rebounds + Assists)

Fantasy scoring model.

⸻

🧠 Smart Projection Engine
	•	Minutes-adjusted stat predictions
	•	Superstar correction layer (prevents star under-projection)
	•	Intelligent turnover fallback logic
	•	Realistic stat bounds
	•	Per-player workload scaling
	•	Rolling window trend modeling

Designed to avoid:
	•	Flat projections
	•	Dead models
	•	Low-minute noise
	•	Underweighting high-usage stars

⸻

📡 Live Sportsbook Scraping

Supports:
	•	PrizePicks (standard + combo props)
	•	Underdog
	•	Sleeper

Automatically normalizes:
	•	Player names
	•	Stat names
	•	Combo formats
	•	Team abbreviations

⸻

📈 Edge Detection

For every prop:
	•	Model projection
	•	Sportsbook line
	•	Edge calculation
	•	Direction (OVER / UNDER)
	•	Minutes context
	•	Source tracking

Outputs:
	•	Unified daily board
	•	Sleeper-only board
	•	Ranked by strongest edges

⸻

🎯 Calibration Layer
	•	Trains a secondary calibration model on real results
	•	Learns sportsbook bias and model bias
	•	Produces “true projection” and “true edge”
	•	Improves long-term ROI

⸻

📉 Performance Tracking

Automatically tracks:
	•	Win rate
	•	Push rate
	•	Average edge
	•	ROI by prop type
	•	ROI by direction
	•	ROI by edge bucket

 🧮 Modeling Approach
	•	Random Forest regressors for stat prediction
	•	Gradient Boosting calibration model
	•	Rolling window feature engineering
	•	Usage-based possession modeling
	•	Per-minute stat normalization
	•	Minutes-weighted regression
	•	Bias correction via post-model calibration

⸻

🎯 Use Cases
	•	Sports trading desks
	•	Prop betting syndicates
	•	Quant sports modeling
	•	Fantasy sports analytics
	•	Market efficiency research
	•	Edge validation & backtesting

⸻

⚠️ Disclaimer

This project is for research and educational purposes only. No guarantees of profitability. Sports betting involves financial risk.

⸻

📬 Contact

Built by: Kylen Dilligard
Focus: Sports Trading, Analytics, Quant Modeling
