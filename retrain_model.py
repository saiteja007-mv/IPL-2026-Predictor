"""
IPL 2026 Model Retrainer
========================
Retrains the XGBoost model from scratch using the current matches.csv.
Can be run standalone or imported by the Streamlit app for in-app retraining.

Usage:
    python retrain_model.py          # Full retrain from matches.csv
    python retrain_model.py --quiet  # Suppress output (for automation)
"""

import os
import sys
import json
import pickle
import zipfile
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Datasets"
MODELS_DIR = BASE_DIR / "Models"

# ── Home grounds (must match training.ipynb) ──────────────────
HOME_GROUNDS = {
    "MI": "Wankhede Stadium", "CSK": "MA Chidambaram Stadium",
    "RCB": "M Chinnaswamy Stadium", "KKR": "Eden Gardens",
    "DC": "Arun Jaitley Stadium", "SRH": "Rajiv Gandhi International Stadium",
    "RR": "Sawai Mansingh Stadium", "PK": "Maharaja Yadavindra Singh Stadium",
    "LSG": "BRSABV Ekana Cricket Stadium", "GT": "Narendra Modi Stadium",
}
SECONDARY_GROUNDS = {
    "RCB": "Shaheed Veer Narayan Singh International Cricket Stadium",
    "PK": "HPCA Cricket Stadium", "RR": "ACA Cricket Stadium",
}

DEFAULT_STATS = {
    "batting_sr": 120.0, "bowling_econ": 8.0,
    "experience": 0, "avg_runs": 20.0, "boundary_pct": 0.5,
}


# ── Helper Functions (mirror training.ipynb) ──────────────────
def update_elo(winner_elo, loser_elo, k=32):
    exp = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    return winner_elo + k * (1 - exp), loser_elo + k * (0 - (1 - exp))


def get_team_form(team, match_history, n=5):
    team_matches = [m for m in match_history if m["team1"] == team or m["team2"] == team]
    recent = team_matches[-n:]
    if not recent:
        return 0.5
    return sum(1 for m in recent if m["winner"] == team) / len(recent)


def get_h2h(team1, team2, match_history):
    h2h = [m for m in match_history
           if (m["team1"] == team1 and m["team2"] == team2) or
              (m["team1"] == team2 and m["team2"] == team1)]
    if not h2h:
        return 0.5
    return sum(1 for m in h2h if m["winner"] == team1) / len(h2h)


def get_venue_stats(team, venue, match_history):
    venue_matches = [m for m in match_history if m["venue"] == venue]
    team_venue = [m for m in venue_matches if m["team1"] == team or m["team2"] == team]
    team_venue_wr = (sum(1 for m in team_venue if m["winner"] == team) / len(team_venue)) if team_venue else 0.5
    if venue_matches:
        bat_first_wins = sum(
            1 for m in venue_matches
            if (m["toss_decision"] == "bat" and m["toss_winner"] == m["winner"])
            or (m["toss_decision"] == "field" and m["toss_winner"] != m["winner"])
        )
        bat_first_pct = bat_first_wins / len(venue_matches)
    else:
        bat_first_pct = 0.5
    return team_venue_wr, bat_first_pct


def is_home(team, venue):
    home = HOME_GROUNDS.get(team, "")
    secondary = SECONDARY_GROUNDS.get(team, "")
    vl = venue.lower()
    if home and home.lower() in vl:
        return 1
    if secondary and secondary.lower() in vl:
        return 1
    return 0


# ── Main Retrain Function ────────────────────────────────────
def retrain(quiet=False):
    """
    Full retrain pipeline. Returns (model, feature_cols, elo_ratings, alias_map, match_history)
    or None on failure.
    """
    if not quiet:
        log.info("Starting full model retrain...")

    # Step 1: Load datasets
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    player_ipl = pd.read_csv(DATA_DIR / "player_ipl_stats.csv")
    player_lifetime = pd.read_csv(DATA_DIR / "player_lifetime_stats.csv")
    team_aliases = pd.read_csv(DATA_DIR / "team_aliases.csv")
    if not quiet:
        log.info(f"Loaded {len(matches)} matches, {len(player_ipl)} IPL players")

    # Step 2: Build alias map
    alias_map = {}
    for _, row in team_aliases.iterrows():
        alias_map[row["alias_name"].strip().lower()] = row["team_alias"].strip()

    def normalize_team(name):
        if pd.isna(name):
            return name
        return alias_map.get(name.strip().lower(), name.strip())

    matches["team1"] = matches["team1"].apply(normalize_team)
    matches["team2"] = matches["team2"].apply(normalize_team)
    matches["winner"] = matches["winner"].apply(normalize_team)
    matches["toss_winner"] = matches["toss_winner"].apply(normalize_team)
    matches = matches[matches["result"] == "win"].reset_index(drop=True)
    if not quiet:
        log.info(f"After filtering no-results: {len(matches)} matches")

    # Step 3: Parse playing XI from cricsheet
    playing_xi_data = {}
    zip_path = DATA_DIR / "cricsheet_cache" / "all_male_json.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            for filename in z.namelist():
                if not filename.endswith(".json"):
                    continue
                data = json.loads(z.read(filename))
                info = data.get("info", {})
                if "Indian Premier League" not in info.get("event", {}).get("name", ""):
                    continue
                mid = int(filename.replace(".json", ""))
                xi = {}
                for tname, plist in info.get("players", {}).items():
                    xi[normalize_team(tname)] = plist
                playing_xi_data[mid] = xi
        if not quiet:
            log.info(f"Extracted playing XI for {len(playing_xi_data)} matches from cricsheet")
    else:
        if not quiet:
            log.warning("No cricsheet zip found — player strength will use defaults for all matches")

    # Step 4: Build player stats lookup
    ipl_stats = {}
    for _, row in player_ipl.iterrows():
        ipl_stats[row["player"]] = {
            "batting_sr": row["batting_sr"] if pd.notna(row["batting_sr"]) else 0,
            "bowling_econ": row["bowling_econ"] if pd.notna(row["bowling_econ"]) else 0,
            "experience": row["ipl_experience"] if pd.notna(row["ipl_experience"]) else 0,
            "avg_runs": row["avg_runs"] if pd.notna(row["avg_runs"]) else 0,
        }
    lifetime_stats = {}
    for _, row in player_lifetime.iterrows():
        lifetime_stats[row["player_name"]] = {
            "batting_sr": row["overall_batting_sr"] if pd.notna(row["overall_batting_sr"]) else 0,
            "bowling_econ": row["overall_bowling_econ"] if pd.notna(row["overall_bowling_econ"]) else 0,
            "experience": row["overall_matches"] if pd.notna(row["overall_matches"]) else 0,
            "avg_runs": row["overall_batting_avg"] if pd.notna(row["overall_batting_avg"]) else 0,
        }

    def get_player_stats(name):
        if name in ipl_stats:
            return ipl_stats[name]
        if name in lifetime_stats:
            return lifetime_stats[name]
        return DEFAULT_STATS.copy()

    def get_team_strength(team_code, match_id):
        xi = playing_xi_data.get(match_id, {}).get(team_code, [])
        if not xi:
            return DEFAULT_STATS.copy()
        stats = [get_player_stats(p) for p in xi]
        return {
            "batting_sr": np.mean([s["batting_sr"] for s in stats]),
            "bowling_econ": np.mean([s["bowling_econ"] for s in stats]),
            "experience": np.mean([s["experience"] for s in stats]),
            "avg_runs": np.mean([s["avg_runs"] for s in stats]),
        }

    # Step 5: Build feature matrix
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    elo_ratings = defaultdict(lambda: 1500)
    match_history = []
    features_list = []

    for _, row in matches.iterrows():
        t1, t2 = row["team1"], row["team2"]
        venue, winner = row["venue"], row["winner"]
        toss_w, toss_d = row["toss_winner"], row["toss_decision"]
        mid = row["match_id"]

        t1_elo, t2_elo = elo_ratings[t1], elo_ratings[t2]
        t1_form = get_team_form(t1, match_history)
        t2_form = get_team_form(t2, match_history)
        h2h_wr = get_h2h(t1, t2, match_history)
        t1_venue, bfp = get_venue_stats(t1, venue, match_history)
        t2_venue, _ = get_venue_stats(t2, venue, match_history)
        t1_str = get_team_strength(t1, mid)
        t2_str = get_team_strength(t2, mid)

        features_list.append({
            "team1_elo": t1_elo, "team2_elo": t2_elo,
            "elo_diff": t1_elo - t2_elo,
            "team1_form": t1_form, "team2_form": t2_form,
            "form_diff": t1_form - t2_form,
            "h2h_team1_wr": h2h_wr,
            "team1_venue_wr": t1_venue, "team2_venue_wr": t2_venue,
            "bat_first_pct": bfp,
            "toss_winner_is_team1": 1 if toss_w == t1 else 0,
            "toss_chose_bat": 1 if toss_d == "bat" else 0,
            "team1_home": is_home(t1, venue),
            "team2_home": is_home(t2, venue),
            "team1_bat_sr": t1_str["batting_sr"],
            "team1_bowl_econ": t1_str["bowling_econ"],
            "team1_experience": t1_str["experience"],
            "team1_avg_runs": t1_str["avg_runs"],
            "team2_bat_sr": t2_str["batting_sr"],
            "team2_bowl_econ": t2_str["bowling_econ"],
            "team2_experience": t2_str["experience"],
            "team2_avg_runs": t2_str["avg_runs"],
            "bat_sr_diff": t1_str["batting_sr"] - t2_str["batting_sr"],
            "bowl_econ_diff": t1_str["bowling_econ"] - t2_str["bowling_econ"],
            "experience_diff": t1_str["experience"] - t2_str["experience"],
            "team1_wins": 1 if winner == t1 else 0,
        })

        match_history.append({
            "team1": t1, "team2": t2, "winner": winner,
            "venue": venue, "toss_winner": toss_w, "toss_decision": toss_d,
        })
        loser = t2 if winner == t1 else t1
        nw, nl = update_elo(elo_ratings[winner], elo_ratings[loser])
        elo_ratings[winner] = nw
        elo_ratings[loser] = nl

    df = pd.DataFrame(features_list)
    if not quiet:
        log.info(f"Feature matrix: {df.shape[0]} x {df.shape[1]}")

    # Step 6: Train XGBoost
    feature_cols = [c for c in df.columns if c != "team1_wins"]
    X = df[feature_cols]
    y = df["team1_wins"]

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss",
    )
    model.fit(X, y)  # Train on ALL data for production
    if not quiet:
        log.info(f"Model trained on {len(X)} matches")

    elo_dict = dict(elo_ratings)

    return model, feature_cols, elo_dict, alias_map, match_history


def save_artifacts(model, feature_cols, elo_ratings, alias_map, match_history, quiet=False):
    """Save all model artifacts to Models/."""
    MODELS_DIR.mkdir(exist_ok=True)
    artifacts = {
        "xgb_model.pkl": model,
        "feature_columns.pkl": feature_cols,
        "elo_ratings.pkl": elo_ratings,
        "alias_map.pkl": alias_map,
        "match_history.pkl": match_history,
    }
    for fname, obj in artifacts.items():
        with open(MODELS_DIR / fname, "wb") as f:
            pickle.dump(obj, f)
    if not quiet:
        log.info(f"Saved {len(artifacts)} artifacts to {MODELS_DIR}")
        for team in sorted(
            [t for t in elo_ratings if t in ["MI","CSK","RCB","DC","KKR","SRH","RR","PK","LSG","GT"]],
            key=lambda t: -elo_ratings[t]
        ):
            log.info(f"  {team}: {elo_ratings[team]:.1f}")


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    result = retrain(quiet=quiet)
    if result:
        model, feature_cols, elo_ratings, alias_map, match_history = result
        save_artifacts(model, feature_cols, elo_ratings, alias_map, match_history, quiet=quiet)
        if not quiet:
            log.info(f"Retrain complete! {len(match_history)} matches, {len(elo_ratings)} teams")
    else:
        log.error("Retrain failed!")
        sys.exit(1)
