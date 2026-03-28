---
name: IPL Match Winner Predictor
overview: Two-notebook system -- training.ipynb (data download, EDA, feature engineering, model training) and prediction.ipynb (load model, predict IPL 2026 matches with analysis). Runs locally with Colab kernel, all data in Datasets/ folder.
todos:
  - id: create-training-notebook
    content: "Create training.ipynb: auto-download data into Datasets/, EDA, feature engineering, train XGBoost + baselines, save model as .pkl"
    status: completed
  - id: create-prediction-notebook
    content: "Create prediction.ipynb: load saved model, input upcoming match details, predict winner with confidence %, feature analysis, and match insights"
    status: completed
isProject: false
---

# IPL 2026 Pre-Match Winner Prediction

Two Jupyter notebooks, both running locally in Cursor with Colab kernel.

## Project Structure

```
IPL 2026/
  Datasets/                    # auto-downloaded by training notebook
    matches.csv                # match-level data (2008-2025)
    deliveries.csv             # ball-by-ball data (2008-2025)
    player_stats.csv           # derived: per-player cumulative IPL career stats
    player_match_stats.csv     # derived: per-player per-match performance
  Models/                      # saved after training
    xgb_ipl_model.pkl          # trained XGBoost model
    label_encoders.pkl         # fitted encoders for categorical features
    feature_columns.pkl        # ordered feature list
  training.ipynb               # Notebook 1: data + EDA + training
  prediction.ipynb             # Notebook 2: predict IPL 2026 matches
```

---

## Notebook 1: training.ipynb

This notebook handles everything from raw data to a saved model.

### Section 1 -- Install & Import

- `pip install` pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, kagglehub, joblib
- Import all libraries

### Section 2 -- Download Data into Datasets/

- Use `kagglehub` to download **maratheabhishek/ipl-dataset-2008-to-2025** (contains `matches.csv` + `deliveries.csv`)
- Copy files into `Datasets/` folder
- Fallback: direct download from GitHub `ritesh-ojha/ipl-dataset` via urllib if kagglehub fails

### Section 3 -- Build Player Stats from deliveries.csv

Instead of relying on a separate player dataset, **derive all player stats directly from `deliveries.csv`**. This gives us full control, covers every player who has ever played IPL (2008-2025), and lets us compute stats up to any date (avoiding data leakage).

**Compute per-player cumulative stats (player_stats.csv):**

Batting stats per player:

- `matches_batted` -- number of innings
- `total_runs` -- career IPL runs
- `batting_avg` -- runs / dismissals
- `batting_sr` -- (runs / balls faced) * 100
- `boundary_pct` -- % of runs from 4s and 6s
- `dot_ball_pct` -- % of balls faced that were dots
- `avg_runs_per_match` -- consistency indicator

Bowling stats per player:

- `matches_bowled` -- number of innings bowled
- `total_wickets` -- career IPL wickets
- `bowling_avg` -- runs conceded / wickets
- `bowling_econ` -- runs conceded per over
- `bowling_sr` -- balls bowled / wickets
- `dot_ball_bowl_pct` -- % of balls that were dots

General:

- `ipl_experience` -- total IPL matches played
- `is_new_player` -- 1 if 0 prior IPL matches (handles new/uncapped players)

**Compute per-player per-match stats (player_match_stats.csv):**

For each match, each player's performance:

- Runs scored, balls faced, strike rate, wickets taken, economy, catches
- Used to compute rolling form (last 3-5 matches) at player level

**Handling new players (critical for IPL 2026):**

- Players with 0 IPL history get league-average stats as defaults
- This prevents NaN issues and gives the model a neutral baseline for unknowns
- As the 2026 season progresses and data accumulates, their real stats replace defaults

### Section 4 -- Load & Explore (EDA)

**matches.csv expected columns:**
`id, city, date, season, match_number, team1, team2, venue, toss_winner, toss_decision, result, dl_applied, winner, win_by_runs, win_by_wickets, player_of_match, umpire1, umpire2`

**deliveries.csv expected columns:**
`id, innings, overs, ball_number, batter, bowler, non_striker, extra_type, batsman_run, extra_run, total_run, is_wicket, player_out, kind, fielders_involved, batting_team`

EDA sections:

- Matches per season, win distribution by team
- Toss impact analysis (does toss winner win more?)
- Venue analysis (batting-first vs chasing win %)
- Team head-to-head heatmap
- Top players by match impact (runs, wickets, strike rate leaders)
- Player experience distribution (how many new vs experienced players per team)
- Batting SR vs bowling economy scatter for all players

### Section 5 -- Feature Engineering

For each historical match, compute these features **using only data available before that match** (no data leakage):

**Team form (rolling window, last 5 matches):**

- `team1_win_rate_last5`, `team2_win_rate_last5`
- `team1_avg_runs_scored_last5`, `team2_avg_runs_scored_last5`

**Head-to-head:**

- `h2h_team1_wins`, `h2h_team2_wins`, `h2h_total_matches`

**Venue stats:**

- `venue_avg_first_innings_score`
- `venue_bat_first_win_pct`
- `venue_team1_win_pct`, `venue_team2_win_pct`

**Toss:**

- `toss_winner_is_team1` (binary)
- `toss_decision` (bat=1, field=0)
- `toss_venue_win_pct` (historical win rate when winning toss at this venue)

**Player-based team strength (aggregated from player_stats + player_match_stats):**

For each team, aggregate over the playing XI (or recent squad) from `player_stats.csv`:

- `team1_avg_batting_sr` -- mean batting strike rate of the team's batters
- `team1_avg_batting_avg` -- mean batting average
- `team1_avg_bowling_econ` -- mean bowling economy of the team's bowlers
- `team1_avg_bowling_sr` -- mean bowling strike rate
- `team1_total_ipl_experience` -- sum of IPL matches played by all XI players
- `team1_new_player_count` -- count of players with 0 prior IPL matches
- `team1_boundary_pct` -- team's average boundary percentage
- (same set for team2)

**Player form (rolling, last 3 matches per player, then aggregated to team):**

- `team1_recent_batting_sr` -- avg of each batter's SR in their last 3 IPL innings
- `team1_recent_bowling_econ` -- avg of each bowler's economy in last 3 IPL spells
- `team1_star_player_form` -- top 3 players' recent performance score
- (same set for team2)

**Context:**

- `season` (encoded)
- `is_playoff` (binary)

**Target:** `team1_wins` (1 if team1 wins, 0 if team2 wins)

### Section 6 -- Model Training

- Drop matches with no result / DLS affected (optional)
- Train/test split: train on 2008-2024, test on 2025 season
- Train three models:
  - **Logistic Regression** (baseline)
  - **Random Forest** (500 trees)
  - **XGBoost** (primary, with hyperparameter tuning via GridSearchCV)
- Evaluation: accuracy, F1, ROC-AUC, confusion matrix
- Feature importance plot from XGBoost
- Save best model + encoders + feature columns to `Models/` using joblib

### Section 7 -- Model Comparison & Summary

- Side-by-side accuracy table
- ROC curves overlay
- Classification report for the best model
- Print final model details

---

## Notebook 2: prediction.ipynb

This notebook loads the trained model and predicts upcoming IPL 2026 matches.

### Section 1 -- Load Model & Data

- Load `xgb_ipl_model.pkl`, `label_encoders.pkl`, `feature_columns.pkl` from `Models/`
- Load all CSVs from `Datasets/` (matches, deliveries, player_stats, player_match_stats)

### Section 2 -- Define Upcoming Match

- User inputs: `team1`, `team2`, `venue`, `toss_winner`, `toss_decision`, `date`
- User inputs: `team1_playing_xi` and `team2_playing_xi` (list of 11 player names each)
- Function computes all features including player-aggregated team strength using each player's cumulative IPL stats and recent form from the player datasets
- New players automatically get league-average defaults

### Section 3 -- Predict & Analyze

- Run prediction: winner + confidence probability
- Show top contributing features (SHAP or XGBoost feature importance for this prediction)
- Display:
  - Head-to-head record between teams
  - Both teams' recent form (last 5 matches)
  - Venue history for both teams
  - Toss impact at this venue
  - **Player matchup analysis**: key batters vs key bowlers from both sides
  - **Team strength comparison**: side-by-side batting SR, bowling economy, experience
  - **New player impact**: flag if either team has debutants and how that historically affects outcomes

### Section 4 -- Batch Predict (Full Matchday)

- Predict all matches for a given IPL 2026 match day
- Output a formatted summary table with winner, confidence, and key factors

### Section 5 -- Update Dataset

- After matches are played, manually add results to `Datasets/matches.csv` and delivery data to `Datasets/deliveries.csv`
- Recompute player stats (player_stats.csv, player_match_stats.csv) to include the new match
- This ensures the next prediction uses updated player form and team data

---

## Key Design Decisions

- **Derive player stats from deliveries.csv, not a separate dataset**: This gives us full control over every player who has played IPL since 2008, lets us compute stats up to any point in time (no leakage), and covers all ~600+ unique players without needing a third-party player database.
- **New player handling**: Players with 0 IPL history (many in IPL 2026 due to mini-auction) get league-average stats as defaults. As the season progresses and they play matches, their real stats replace defaults. This prevents NaN issues and gives the model a neutral baseline.
- **Player data aggregated to team level**: Individual player stats are aggregated into team-level features (avg batting SR, avg bowling economy, total experience, etc.) for the model. This keeps the feature space manageable while capturing squad quality differences.
- **XGBoost over deep learning**: For tabular pre-match features, XGBoost matches or beats neural networks and is far simpler. Deep learning (LSTM/GRU) only benefits ball-by-ball live prediction, which is out of scope.
- **Rolling features at both team and player level**: Team form (last 5 matches) + individual player form (last 3 innings) captures current momentum, which is critical in T20 cricket.
- **Strict temporal ordering**: Features for each match are computed using only past data to prevent data leakage, which is the most common mistake in cricket prediction models.
- **Retrain as season progresses**: After adding IPL 2026 results, re-run training.ipynb to adapt the model to current season conditions and new player data.

