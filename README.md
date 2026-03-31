# IPL 2026 Match Winner Predictor

An ML-powered match prediction system for the Indian Premier League 2026 season, trained on **18 seasons** of historical IPL data (2008–2025, 1,169 matches).

**Live App:** [techrex-ipl-prediction.streamlit.app](https://techrex-ipl-prediction.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)

---

## How It Works

1. **Select two teams**, venue, toss winner, and toss decision
2. **Pick Playing XI** for each team from the 2026 squad list (with overseas player limit tracking — max 4 per team)
3. **Pick the projected impact player** from the bench for each team
4. **Get a prediction** with win probability, radar chart, factor scorecard, and a plain-English breakdown of why

## Features

- **25-feature XGBoost model** — Elo ratings, recent form, head-to-head record, venue stats, toss impact, and player-level batting/bowling/experience metrics
- **Smart name resolution** — enter player names in any format (full name, cricsheet format, or partial) and the system resolves them automatically
- **Overseas player tracking** — flags overseas players and enforces the 4-per-XI limit
- **Impact-player aware predictions** — projects one bench substitute per team, enforces the overseas restriction for that rule, and adjusts the effective batting or bowling strength based on toss context
- **Visual breakdowns** — radar chart comparing 6 dimensions, factor scorecard, and comparison bars for every metric
- **Excel tracker** — `output/IPL_2026_Predictions.xlsx` with dropdown menus for teams/venues, dependent dropdowns for toss winner and match result, conditional formatting (green = correct, red = wrong), and a summary sheet with accuracy stats

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBoost (gradient-boosted trees) |
| Training Data | 1,169 IPL matches (2008–2025) |
| Features | 25 (team strength, form, H2H, venue, toss, player stats) |
| Split | 80/20 random split |
| Hyperparameters | 200 estimators, max depth 5, learning rate 0.05 |

### Feature Categories

- **Team Strength** — Elo rating, Elo difference
- **Form** — Win % in last 5 matches, form difference
- **Head-to-Head** — Historical win rate between the two teams
- **Venue** — Each team's win rate at the venue, bat-first win %
- **Toss** — Toss winner, bat/field decision
- **Home Advantage** — Whether either team is playing at their home ground
- **Player Stats** — Average batting strike rate, bowling economy, experience, and runs per player in the Playing XI

### Impact Player Modeling

The saved XGBoost model still uses the same 25 features. The impact-player rule is modeled at prediction time by adjusting the effective lineup strength before those features are assembled:

- If a team is **batting first**, the app assumes the impact substitute is most likely to matter while **defending**, so it looks for a bowling upgrade from the bench.
- If a team is **chasing**, the app assumes the impact substitute is most likely to matter in the **run chase**, so it looks for a batting upgrade from the bench.
- If the starting XI already has **4 overseas players**, the projected impact player must be Indian.

This keeps the trained model compatible with the existing artifacts while letting 2026 predictions react to the impact-player rule.

## Project Structure

```
.
├── streamlit_app.py              # Streamlit web app (main entry point)
├── training.ipynb                # ML pipeline: data processing, feature engineering, model training
├── requirements.txt              # Python dependencies
├── CLAUDE.md                     # AI assistant context file
├── Datasets/
│   ├── matches.csv               # Match results (2008–2025)
│   ├── ball_by_ball_data.csv     # Ball-by-ball summary stats
│   ├── player_ipl_stats.csv      # Per-player IPL career stats
│   ├── player_lifetime_stats.csv # Cross-league T20 career stats
│   ├── players-data-updated.csv  # Player metadata (bat/bowl style, full names)
│   ├── team_aliases.csv          # Team name normalization map
│   ├── ipl_2026_squads.json      # Full 2026 squad lists (10 teams, 25 players each)
│   └── ipl_2026_name_map.csv     # Name mappings for new/uncapped players
├── Models/
│   ├── xgb_model.pkl             # Trained XGBoost model
│   ├── elo_ratings.pkl           # Team Elo ratings
│   ├── feature_columns.pkl       # Feature column order
│   ├── alias_map.pkl             # Team alias lookup
│   └── match_history.pkl         # Historical match records for form/H2H
├── IPL Team logos/               # Team logo PNGs (original + processed)
└── output/
    └── IPL_2026_Predictions.xlsx # Prediction tracker (auto-generated)
```

## Setup

### Run Locally

```bash
# Clone the repo
git clone https://github.com/saiteja007-mv/IPL-2026-Predictor.git
cd IPL-2026-Predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

### Retrain the Model

Open `training.ipynb` in Jupyter and run all cells. This will:
1. Load and clean match data from `Datasets/`
2. Normalize team names via `team_aliases.csv`
3. Build Elo ratings, form, H2H, venue stats, and player-level features
4. Train an XGBoost classifier
5. Save model artifacts to `Models/`

## IPL 2026 Teams

| Code | Team | Captain | Home Ground |
|------|------|---------|-------------|
| MI | Mumbai Indians | Hardik Pandya | Wankhede Stadium, Mumbai |
| CSK | Chennai Super Kings | Ruturaj Gaikwad | MA Chidambaram Stadium, Chennai |
| RCB | Royal Challengers Bengaluru | Rajat Patidar | M Chinnaswamy Stadium, Bengaluru |
| KKR | Kolkata Knight Riders | Ajinkya Rahane | Eden Gardens, Kolkata |
| DC | Delhi Capitals | Axar Patel | Arun Jaitley Stadium, Delhi |
| SRH | Sunrisers Hyderabad | Pat Cummins | Rajiv Gandhi Intl Stadium, Hyderabad |
| RR | Rajasthan Royals | Riyan Parag | Sawai Mansingh Stadium, Jaipur |
| PK | Punjab Kings | Shreyas Iyer | Maharaja Yadavindra Singh Stadium, Mullanpur |
| LSG | Lucknow Super Giants | Rishabh Pant | Ekana Cricket Stadium, Lucknow |
| GT | Gujarat Titans | Shubman Gill | Narendra Modi Stadium, Ahmedabad |

## Data Sources

- Match data: [Cricsheet](https://cricsheet.org/) (ball-by-ball JSON archives)
- Player stats: Compiled from Cricsheet delivery data + ESPN Cricinfo
- Squad info: Deccan Herald, ESPN Cricinfo, Wikipedia, official IPL sources (March 2026)

## License

This project is for educational and personal use. IPL is a registered trademark of BCCI.
