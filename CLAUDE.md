# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IPL 2026 match winner prediction system trained on 18 seasons of historical IPL data (2008–2025, 1,169 matches). Predicts match outcomes via a 25-feature XGBoost model with a Streamlit web UI.

**Live App:** [techrex-ipl-prediction.streamlit.app](https://techrex-ipl-prediction.streamlit.app/)

## Commands

```bash
# Run the web app
streamlit run streamlit_app.py

# Install dependencies
pip install -r requirements.txt

# Retrain the model (run all cells in Jupyter)
jupyter notebook training.ipynb
```

## Architecture

### Entry Points

- **`streamlit_app.py`** — Single-file Streamlit app (~1700 lines). Contains all UI, prediction logic, and helper functions. No separate modules.
- **`training.ipynb`** — ML pipeline: data cleaning → feature engineering → XGBoost training → artifact export to `Models/`.

### Prediction Flow

1. User selects teams, venue, toss, Playing XI (max 4 overseas), and impact player from bench
2. `build_effective_team_strength()` applies the impact substitution to adjust lineup stats before feature assembly
3. `predict_match()` assembles the 25-feature vector (Elo, form, H2H, venue, toss, player stats) and calls the loaded XGBoost model
4. Results are shown with radar chart, factor scorecard, and probability bars

### Key Functions in `streamlit_app.py`

| Function | Purpose |
|----------|---------|
| `load_model()` | Loads all 5 pkl artifacts from `Models/` |
| `load_player_stats()` | Loads CSVs + squads JSON |
| `build_name_resolver()` | Fuzzy name matching across all naming formats |
| `apply_impact_substitution()` | Swaps in impact player, adjusting batting/bowling strength |
| `build_effective_team_strength()` | Determines phase (bat/bowl) and returns adjusted XI stats |
| `predict_match()` | Assembles feature vector and returns win probabilities |
| `render_xi_selector()` | Renders the Playing XI selection UI with overseas tracking |

### Model Artifacts (`Models/`)

All 5 files must exist before the app can run. They are produced by `training.ipynb`:

| File | Contents |
|------|---------|
| `xgb_model.pkl` | Trained XGBoost classifier (25 features) |
| `elo_ratings.pkl` | Team Elo ratings keyed by canonical team code |
| `feature_columns.pkl` | Ordered list of feature names (must match at prediction time) |
| `alias_map.pkl` | Team name → canonical code lookup |
| `match_history.pkl` | List of historical match dicts for form/H2H/venue lookups |

## Datasets

All source data is in `Datasets/`:

| File | Description |
|------|-------------|
| `matches.csv` | Match results 2008–2025 (`match_id, season_id, city, date, venue, toss_winner, team1, team2, toss_decision, winner, win_by_runs, win_by_wickets, player_of_match, result`) |
| `ball_by_ball_data.csv` | Aggregated ball-by-ball summary stats (~278K deliveries) |
| `player_ipl_stats.csv` | Per-player IPL career aggregates (`player, matches_batted, total_runs, balls_faced, fours, sixes, batting_sr, avg_runs, boundary_pct, matches_bowled, balls_bowled, runs_conceded, wickets, overs, bowling_econ, bowling_sr, ipl_experience`) |
| `player_lifetime_stats.csv` | Cross-league T20 career stats (`player_id, player_name, bat_style, bowl_style` + per-league stats) — fallback for uncapped/new players |
| `players-data-updated.csv` | Player metadata: bat/bowl style, field position, full name, player_id |
| `team_aliases.csv` | Maps all historical team name variants to canonical short codes |
| `ipl_2026_squads.json` | Full 2026 squad lists (10 teams, 25 players each) with overseas flags |
| `ipl_2026_name_map.csv` | Name mappings for new/uncapped players not in historical stats |

## Design Principles

- Always resolve team names through `team_aliases.csv` — never hardcode variants. Covers defunct teams (RPS, GL, KTK, PW) and spelling variants.
- Prefer `player_ipl_stats.csv` for player stats; fall back to `player_lifetime_stats.csv` for uncapped players.
- The XGBoost model has a fixed 25-feature contract. The impact-player rule is applied **before** feature assembly (in `build_effective_team_strength()`), not by changing the model.
- Append new 2026 match results to `matches.csv` after each game; re-run `training.ipynb` to rebuild Elo/form and retrain.
- If adding new datasets or scripts, add path constants to a centralised location rather than scattering literals.

## IPL 2026 Teams

| Code | Team | Captain | Home Ground |
|------|------|---------|-------------|
| MI | Mumbai Indians | Hardik Pandya | Wankhede Stadium |
| CSK | Chennai Super Kings | Ruturaj Gaikwad | MA Chidambaram Stadium |
| RCB | Royal Challengers Bengaluru | Rajat Patidar | M Chinnaswamy Stadium |
| KKR | Kolkata Knight Riders | Ajinkya Rahane | Eden Gardens |
| DC | Delhi Capitals | Axar Patel | Arun Jaitley Stadium |
| SRH | Sunrisers Hyderabad | Pat Cummins | Rajiv Gandhi International Stadium |
| RR | Rajasthan Royals | Riyan Parag | Sawai Mansingh Stadium |
| PK | Punjab Kings | Shreyas Iyer | Maharaja Yadavindra Singh Stadium |
| LSG | Lucknow Super Giants | Rishabh Pant | BRSABV Ekana Cricket Stadium |
| GT | Gujarat Titans | Shubman Gill | Narendra Modi Stadium |
