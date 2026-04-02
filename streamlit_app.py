"""
IPL 2026 Match Predictor — Streamlit Web App

Run: streamlit run app.py
Requires: trained model artifacts in Models/ (run training.ipynb first)
"""

import os
import re
import html
import json
import pickle
import sqlite3
import hashlib
import secrets
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
from difflib import SequenceMatcher
from textwrap import dedent

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="IPL 2026 Predictor",
    page_icon="🏏",
    layout="wide",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    /* Dark cricket theme */
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff6b35, #f7c948);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-top: 0;
    }
    .winner-box {
        text-align: center;
        padding: 30px;
        border-radius: 16px;
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        border: 2px solid #52b788;
        margin: 20px 0;
    }
    .winner-name {
        font-size: 2rem;
        font-weight: 800;
        color: #d4af37;
    }
    .winner-label {
        font-size: 0.9rem;
        color: #95d5b2;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    .stat-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
    }
    .prob-bar {
        height: 40px;
        border-radius: 8px;
        display: flex;
        overflow: hidden;
        margin: 10px 0;
    }
    .prob-t1 {
        background: linear-gradient(90deg, #2196F3, #42A5F5);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .prob-t2 {
        background: linear-gradient(90deg, #FF5722, #FF7043);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .insight-card {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    .insight-title {
        color: #8b949e;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .insight-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e6edf3;
    }
    .page-section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 4px;
    }
    .page-section-subtitle {
        color: #8b949e;
        font-size: 0.92rem;
        margin-bottom: 14px;
    }
    .match-overview {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 10px 0 18px;
    }
    .match-chip {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 12px 14px;
    }
    .match-chip-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        color: #8b949e;
        margin-bottom: 6px;
    }
    .match-chip-value {
        font-size: 1rem;
        font-weight: 600;
        color: #e6edf3;
        line-height: 1.35;
    }
    .selector-stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin: 12px 0 14px;
    }
    .selector-stat {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 12px 14px;
    }
    .selector-stat-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.85px;
        color: #8b949e;
        margin-bottom: 6px;
    }
    .selector-stat-value {
        font-size: 1rem;
        font-weight: 600;
        color: #e6edf3;
        line-height: 1.35;
    }
    .section-kicker {
        color: #8b949e;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    @media (max-width: 900px) {
        .match-overview,
        .selector-stats {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

APP_DB_PATH = ".streamlit/ipl2026_app.db"
PASSWORD_HASH_ROUNDS = 200_000
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,24}$")


def current_timestamp():
    """Return a consistent local timestamp for saved app records."""
    return datetime.now().isoformat(timespec="seconds")


def get_db_connection():
    """Open the local app database with row access."""
    os.makedirs(os.path.dirname(APP_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_app_database():
    """Create local tables for auth, saved XIs, and prediction history."""
    with get_db_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS xi_combinations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                team_code TEXT NOT NULL,
                team_name TEXT NOT NULL,
                combo_name TEXT NOT NULL,
                playing_xi_json TEXT NOT NULL,
                impact_player TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, team_code, combo_name),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                team1_name TEXT NOT NULL,
                team1_code TEXT NOT NULL,
                team2_name TEXT NOT NULL,
                team2_code TEXT NOT NULL,
                venue TEXT NOT NULL,
                toss_winner TEXT NOT NULL,
                toss_decision TEXT NOT NULL,
                team1_xi_json TEXT NOT NULL,
                team2_xi_json TEXT NOT NULL,
                team1_impact TEXT,
                team2_impact TEXT,
                winner_name TEXT NOT NULL,
                winner_code TEXT NOT NULL,
                winner_prob REAL NOT NULL,
                team1_prob REAL NOT NULL,
                team2_prob REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_xi_combinations_user_team
            ON xi_combinations(user_id, team_code, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_prediction_history_user_time
            ON prediction_history(user_id, created_at DESC);
            """
        )


def ensure_app_session_state():
    """Seed Streamlit session keys used by auth and flash messages."""
    st.session_state.setdefault("auth_user", None)


def set_flash_message(kind, message):
    """Store a one-shot UI message for the next rerun."""
    st.session_state["flash_message"] = {"kind": kind, "message": message}


def render_flash_message():
    """Render and clear the latest flash message, if any."""
    flash = st.session_state.pop("flash_message", None)
    if not flash:
        return
    getattr(st, flash["kind"], st.info)(flash["message"])


def normalize_username(username):
    """Normalize usernames for storage and lookup."""
    return username.strip().lower()


def hash_password(password, salt_hex):
    """Derive a password hash using PBKDF2."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        PASSWORD_HASH_ROUNDS,
    ).hex()


def validate_new_user(username, password, confirm_password):
    """Validate a new account payload before inserting it."""
    normalized = normalize_username(username)
    if not USERNAME_PATTERN.fullmatch(normalized):
        return "Username must be 3-24 characters and use only letters, numbers, ., _, or -."
    if len(password) < 6:
        return "Password must be at least 6 characters."
    if password != confirm_password:
        return "Passwords do not match."
    return None


def register_user(username, password):
    """Create a local user account and return the saved identity."""
    normalized = normalize_username(username)
    salt_hex = secrets.token_hex(16)
    password_hash = hash_password(password, salt_hex)
    created_at = current_timestamp()

    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, password_hash, salt, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (normalized, password_hash, salt_hex, created_at),
            )
        return {"id": cursor.lastrowid, "username": normalized}, None
    except sqlite3.IntegrityError:
        return None, "That username is already registered on this device."


def authenticate_user(username, password):
    """Validate local credentials and return the account record."""
    normalized = normalize_username(username)
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT id, username, password_hash, salt
            FROM users
            WHERE username = ?
            """,
            (normalized,),
        ).fetchone()

    if not row:
        return None

    candidate_hash = hash_password(password, row["salt"])
    if not secrets.compare_digest(candidate_hash, row["password_hash"]):
        return None

    return {"id": row["id"], "username": row["username"]}


def save_xi_combination(user_id, team_code, team_name, combo_name, playing_xi, impact_player):
    """Insert or update a named Playing XI combination for a user."""
    combo_name = combo_name.strip()
    timestamp = current_timestamp()

    with get_db_connection() as conn:
        existing = conn.execute(
            """
            SELECT id
            FROM xi_combinations
            WHERE user_id = ? AND team_code = ? AND combo_name = ?
            """,
            (user_id, team_code, combo_name),
        ).fetchone()

        conn.execute(
            """
            INSERT INTO xi_combinations (
                user_id, team_code, team_name, combo_name,
                playing_xi_json, impact_player, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, team_code, combo_name)
            DO UPDATE SET
                team_name = excluded.team_name,
                playing_xi_json = excluded.playing_xi_json,
                impact_player = excluded.impact_player,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                team_code,
                team_name,
                combo_name,
                json.dumps(playing_xi, ensure_ascii=False),
                impact_player,
                timestamp,
                timestamp,
            ),
        )

    return "updated" if existing else "saved"


def delete_xi_combination(user_id, combination_id):
    """Delete a saved XI combination owned by the current user."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            DELETE FROM xi_combinations
            WHERE id = ? AND user_id = ?
            """,
            (combination_id, user_id),
        )
    return cursor.rowcount > 0


def list_user_xi_combinations(user_id, team_code=None):
    """Return saved XI combinations for the current user."""
    if not user_id:
        return []

    query = """
        SELECT id, team_code, team_name, combo_name,
               playing_xi_json, impact_player, created_at, updated_at
        FROM xi_combinations
        WHERE user_id = ?
    """
    params = [user_id]
    if team_code:
        query += " AND team_code = ?"
        params.append(team_code)
    query += " ORDER BY updated_at DESC, combo_name ASC"

    with get_db_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    combinations = []
    for row in rows:
        try:
            playing_xi = json.loads(row["playing_xi_json"])
        except json.JSONDecodeError:
            playing_xi = []
        combinations.append(
            {
                "id": row["id"],
                "team_code": row["team_code"],
                "team_name": row["team_name"],
                "combo_name": row["combo_name"],
                "playing_xi": playing_xi,
                "impact_player": row["impact_player"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    return combinations


def get_user_xi_combinations_for_team(user_id, team_code, squad):
    """Return saved combinations, annotated for the current squad."""
    combinations = list_user_xi_combinations(user_id, team_code=team_code)
    enriched = []

    for combo in combinations:
        filtered_xi = [player for player in combo["playing_xi"] if player in squad][:11]
        missing_players = [player for player in combo["playing_xi"] if player not in squad]
        valid_impact = combo["impact_player"]
        if valid_impact not in squad or valid_impact in filtered_xi:
            valid_impact = None

        enriched.append(
            {
                **combo,
                "filtered_xi": filtered_xi,
                "missing_players": missing_players,
                "valid_impact_player": valid_impact,
            }
        )

    return enriched


def load_combination_into_state(xi_widget_key, selected_display, impact_widget_key, impact_display, combo_name_key, combo_name):
    """Load a saved XI combination into the active widget state."""
    st.session_state[xi_widget_key] = selected_display
    st.session_state[impact_widget_key] = impact_display
    st.session_state[combo_name_key] = combo_name


def save_prediction_history(user_id, prediction_payload):
    """Persist a logged-in user's prediction and selected squads."""
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO prediction_history (
                user_id, team1_name, team1_code, team2_name, team2_code,
                venue, toss_winner, toss_decision,
                team1_xi_json, team2_xi_json, team1_impact, team2_impact,
                winner_name, winner_code, winner_prob,
                team1_prob, team2_prob, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                prediction_payload["team1_name"],
                prediction_payload["team1_code"],
                prediction_payload["team2_name"],
                prediction_payload["team2_code"],
                prediction_payload["venue"],
                prediction_payload["toss_winner"],
                prediction_payload["toss_decision"],
                json.dumps(prediction_payload["team1_xi"], ensure_ascii=False),
                json.dumps(prediction_payload["team2_xi"], ensure_ascii=False),
                prediction_payload["team1_impact"],
                prediction_payload["team2_impact"],
                prediction_payload["winner_name"],
                prediction_payload["winner_code"],
                float(prediction_payload["winner_prob"]),
                float(prediction_payload["team1_prob"]),
                float(prediction_payload["team2_prob"]),
                current_timestamp(),
            ),
        )


def delete_prediction_history_entry(user_id, prediction_id):
    """Delete a saved prediction owned by the current user."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            DELETE FROM prediction_history
            WHERE id = ? AND user_id = ?
            """,
            (prediction_id, user_id),
        )
    return cursor.rowcount > 0


def get_user_prediction_history(user_id, limit=20):
    """Return the most recent saved predictions for a user."""
    if not user_id:
        return []

    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    history = []
    for row in rows:
        try:
            team1_xi = json.loads(row["team1_xi_json"])
        except json.JSONDecodeError:
            team1_xi = []
        try:
            team2_xi = json.loads(row["team2_xi_json"])
        except json.JSONDecodeError:
            team2_xi = []

        history.append(
            {
                "id": row["id"],
                "team1_name": row["team1_name"],
                "team1_code": row["team1_code"],
                "team2_name": row["team2_name"],
                "team2_code": row["team2_code"],
                "venue": row["venue"],
                "toss_winner": row["toss_winner"],
                "toss_decision": row["toss_decision"],
                "team1_xi": team1_xi,
                "team2_xi": team2_xi,
                "team1_impact": row["team1_impact"],
                "team2_impact": row["team2_impact"],
                "winner_name": row["winner_name"],
                "winner_code": row["winner_code"],
                "winner_prob": row["winner_prob"],
                "team1_prob": row["team1_prob"],
                "team2_prob": row["team2_prob"],
                "created_at": row["created_at"],
            }
        )
    return history


def get_user_dashboard_counts(user_id):
    """Return quick counts for saved predictions and XI combinations."""
    if not user_id:
        return {"predictions": 0, "combinations": 0}

    with get_db_connection() as conn:
        prediction_count = conn.execute(
            "SELECT COUNT(*) FROM prediction_history WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]
        combination_count = conn.execute(
            "SELECT COUNT(*) FROM xi_combinations WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]

    return {"predictions": prediction_count, "combinations": combination_count}

# ============================================================
# Team Data
# ============================================================
TEAMS = {
    "Mumbai Indians": "MI",
    "Chennai Super Kings": "CSK",
    "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders": "KKR",
    "Delhi Capitals": "DC",
    "Sunrisers Hyderabad": "SRH",
    "Rajasthan Royals": "RR",
    "Punjab Kings": "PK",
    "Lucknow Super Giants": "LSG",
    "Gujarat Titans": "GT",
}

VENUES = [
    "Wankhede Stadium",
    "MA Chidambaram Stadium",
    "M Chinnaswamy Stadium",
    "Eden Gardens",
    "Arun Jaitley Stadium",
    "Rajiv Gandhi International Stadium",
    "Sawai Mansingh Stadium",
    "Maharaja Yadavindra Singh Stadium",
    "BRSABV Ekana Cricket Stadium",
    "Narendra Modi Stadium",
    "Shaheed Veer Narayan Singh International Cricket Stadium",
    "HPCA Cricket Stadium",
    "ACA Cricket Stadium",
]

HOME_GROUNDS = {
    "MI": "Wankhede Stadium",
    "CSK": "MA Chidambaram Stadium",
    "RCB": "M Chinnaswamy Stadium",
    "KKR": "Eden Gardens",
    "DC": "Arun Jaitley Stadium",
    "SRH": "Rajiv Gandhi International Stadium",
    "RR": "Sawai Mansingh Stadium",
    "PK": "Maharaja Yadavindra Singh Stadium",
    "LSG": "BRSABV Ekana Cricket Stadium",
    "GT": "Narendra Modi Stadium",
}

# Secondary home venues (also count as "home" for these teams)
SECONDARY_GROUNDS = {
    "RCB": "Shaheed Veer Narayan Singh International Cricket Stadium",
    "PK": "HPCA Cricket Stadium",
    "RR": "ACA Cricket Stadium",
}

# Overseas (non-Indian) players per team — max 4 allowed in playing XI
OVERSEAS_PLAYERS = {
    "MI": {"Ryan Rickelton", "Mitchell Santner", "Corbin Bosch", "Trent Boult", "Allah Ghafanzar", "Will Jacks", "Sherfane Rutherford", "Quinton de Kock"},
    "CSK": {"Dewald Brevis", "Jamie Overton", "Noor Ahmad", "Nathan Ellis", "Akeal Hosein", "Matthew Short", "Matt Henry", "Zak Foulkes"},
    "RCB": {"Phil Salt", "Tim David", "Romario Shepherd", "Jacob Bethell", "Josh Hazlewood", "Nuwan Thushara", "Jacob Duffy", "Jordan Cox"},
    "KKR": {"Rovman Powell", "Sunil Narine", "Cameron Green", "Finn Allen", "Matheesha Pathirana", "Tim Seifert", "Rachin Ravindra", "Blessing Muzarabani"},
    "DC": {"Dushmantha Chameera", "Mitchell Starc", "Tristan Stubbs", "David Miller", "Ben Duckett", "Pathum Nissanka", "Lungi Ngidi", "Kyle Jamieson"},
    "SRH": {"Pat Cummins", "Travis Head", "Heinrich Klaasen", "Kamindu Mendis", "Brydon Carse", "Liam Livingstone", "David Payne"},
    "RR": {"Sam Curran", "Donovan Ferreira", "Lhuan-Dre Pretorius", "Shimron Hetmyer", "Jofra Archer", "Kwena Maphaka", "Nandre Burger", "Adam Milne"},
    "PK": {"Marcus Stoinis", "Azmatullah Omarzai", "Marco Jansen", "Mitchell Owen", "Xavier Bartlett", "Lockie Ferguson", "Cooper Connolly", "Ben Dwarshuis"},
    "LSG": {"Aiden Markram", "Matthew Breetzke", "Nicholas Pooran", "Mitchell Marsh", "Wanindu Hasaranga", "Anrich Nortje", "Josh Inglis"},
    "GT": {"Jos Buttler", "Glenn Phillips", "Kagiso Rabada", "Rashid Khan", "Jason Holder", "Tom Banton", "Luke Wood"},
}

LOGO_DIR = "IPL Team logos/processed"

MANUAL_PLAYER_STATS_OVERRIDES = {
    # Official IPL profile describes David Payne as an overseas bowler with 230+ T20 appearances.
    "David Payne": {
        "batting_sr": 85.0,
        "bowling_econ": 8.0,
        "experience": 233.0,
        "avg_runs": 6.0,
        "batting_innings": 35.0,
        "bowling_innings": 233.0,
        "balls_bowled": 5592.0,
        "wickets": 304.0,
    },
}

MANUAL_PLAYER_META_OVERRIDES = {
    "david payne": {
        "bat_style": "Right hand Bat",
        "bowl_style": "Left arm Medium fast",
    },
}


# ============================================================
# Load Model & Artifacts (cached so it loads only once)
# ============================================================
@st.cache_resource
def load_model():
    """Load all saved model artifacts."""
    models_dir = "Models"
    with open(f"{models_dir}/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{models_dir}/feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    with open(f"{models_dir}/elo_ratings.pkl", "rb") as f:
        elo_ratings = pickle.load(f)
    with open(f"{models_dir}/alias_map.pkl", "rb") as f:
        alias_map = pickle.load(f)
    with open(f"{models_dir}/match_history.pkl", "rb") as f:
        match_history = pickle.load(f)
    return model, feature_cols, elo_ratings, alias_map, match_history


@st.cache_resource
def load_player_stats():
    """Load player stats for lookup."""
    player_ipl = pd.read_csv("Datasets/player_ipl_stats.csv")
    player_lifetime = pd.read_csv("Datasets/player_lifetime_stats.csv")

    ipl_stats = {}
    for _, row in player_ipl.iterrows():
        ipl_stats[row["player"]] = {
            "batting_sr": row["batting_sr"] if pd.notna(row["batting_sr"]) else 0,
            "bowling_econ": row["bowling_econ"] if pd.notna(row["bowling_econ"]) else 0,
            "experience": row["ipl_experience"] if pd.notna(row["ipl_experience"]) else 0,
            "avg_runs": row["avg_runs"] if pd.notna(row["avg_runs"]) else 0,
            "batting_innings": row["matches_batted"] if pd.notna(row["matches_batted"]) else 0,
            "bowling_innings": row["matches_bowled"] if pd.notna(row["matches_bowled"]) else 0,
            "balls_bowled": row["balls_bowled"] if pd.notna(row["balls_bowled"]) else 0,
            "wickets": row["wickets"] if pd.notna(row["wickets"]) else 0,
        }

    lifetime = {}
    for _, row in player_lifetime.iterrows():
        lifetime[row["player_name"]] = {
            "batting_sr": row["overall_batting_sr"] if pd.notna(row["overall_batting_sr"]) else 0,
            "bowling_econ": row["overall_bowling_econ"] if pd.notna(row["overall_bowling_econ"]) else 0,
            "experience": row["overall_matches"] if pd.notna(row["overall_matches"]) else 0,
            "avg_runs": row["overall_batting_avg"] if pd.notna(row["overall_batting_avg"]) else 0,
            "batting_innings": row["overall_batting_innings"] if pd.notna(row["overall_batting_innings"]) else 0,
            "bowling_innings": row["overall_bowling_innings"] if pd.notna(row["overall_bowling_innings"]) else 0,
            "balls_bowled": row["overall_balls_bowled"] if pd.notna(row["overall_balls_bowled"]) else 0,
            "wickets": row["overall_wickets"] if pd.notna(row["overall_wickets"]) else 0,
        }

    for player_name, stats in MANUAL_PLAYER_STATS_OVERRIDES.items():
        lifetime[player_name] = stats.copy()

    return ipl_stats, lifetime


@st.cache_resource
def build_name_resolver():
    """
    Build mappings to resolve full player names (e.g. "Virat Kohli")
    to cricsheet format (e.g. "V Kohli").

    Strategy (in order):
    1. Exact match in ipl_stats / lifetime_stats — already works, skip
    2. Full name lookup from players-data-updated.csv
    3. Fuzzy match against all known names
    """
    meta = pd.read_csv("Datasets/players-data-updated.csv")

    # Map: lowercase full name -> cricsheet name
    # e.g. "virat kohli" -> "V Kohli"
    full_to_cricsheet = {}
    for _, row in meta.iterrows():
        cricsheet = row["player_name"]
        full_name = row["player_full_name"]
        if pd.notna(full_name) and pd.notna(cricsheet):
            # Map the full name exactly
            full_to_cricsheet[full_name.strip().lower()] = cricsheet

            # Also map "FirstName LastName" combinations from multi-part names
            # e.g. "Jasprit Jasbirsingh Bumrah" -> also map "Jasprit Bumrah"
            parts = full_name.strip().split()
            if len(parts) >= 3:
                # First + Last: "Jasprit Bumrah"
                full_to_cricsheet[f"{parts[0]} {parts[-1]}".lower()] = cricsheet
                # Any middle name + Last: e.g. "Wanindu de Silva" or "Wanindu Hasaranga"
                for mid in parts[1:-1]:
                    full_to_cricsheet[f"{parts[0]} {mid}".lower()] = cricsheet
                    full_to_cricsheet[f"{mid} {parts[-1]}".lower()] = cricsheet
                # For very long names, try all pairs of consecutive words
                if len(parts) >= 4:
                    for i in range(len(parts) - 1):
                        full_to_cricsheet[f"{parts[i]} {parts[i+1]}".lower()] = cricsheet

            # Map cricsheet name itself (in case user types it)
            full_to_cricsheet[cricsheet.strip().lower()] = cricsheet

    # Load IPL 2026 specific name mappings (covers new/uncapped players)
    name_map_path = "Datasets/ipl_2026_name_map.csv"
    if os.path.exists(name_map_path):
        name_map = pd.read_csv(name_map_path)
        for _, row in name_map.iterrows():
            full_to_cricsheet[row["full_name"].strip().lower()] = row["cricsheet_name"].strip()

    return full_to_cricsheet, meta


# ============================================================
# Helper Functions (same logic as training notebook)
# ============================================================
ROLE_ICONS = {
    "batter": "🏏",
    "bowler": "🎯",
    "all_rounder": "🧰",
}

ROLE_LABELS = {
    "batter": "batter",
    "bowler": "bowler",
    "all_rounder": "all-rounder",
}


def build_player_meta_lookup(players_meta):
    """Create a lowercase name -> batting/bowling style lookup."""
    lookup = {}
    for _, row in players_meta.iterrows():
        bat_style = row["bat_style"] if pd.notna(row["bat_style"]) else ""
        bowl_style = row["bowl_style"] if pd.notna(row["bowl_style"]) else ""

        keys = set()
        for col in ("player_name", "player_full_name"):
            value = row.get(col)
            if pd.notna(value) and str(value).strip():
                keys.add(str(value).strip().lower())

        full_name = row["player_full_name"] if "player_full_name" in row and pd.notna(row["player_full_name"]) else ""
        if full_name:
            parts = full_name.strip().split()
            if len(parts) >= 2:
                keys.add(f"{parts[0]} {parts[-1]}".lower())

        for key in keys:
            lookup[key] = {
                "bat_style": bat_style,
                "bowl_style": bowl_style,
            }

    lookup.update(MANUAL_PLAYER_META_OVERRIDES)
    return lookup


DEFAULT_STATS = {
    "batting_sr": 120.0,
    "bowling_econ": 8.0,
    "experience": 0,
    "avg_runs": 20.0,
    "batting_innings": 0,
    "bowling_innings": 0,
    "balls_bowled": 0,
    "wickets": 0,
}


def resolve_player_name(name, full_to_cricsheet, all_cricsheet_names):
    """
    Resolve a user-typed name to the cricsheet format name.
    Returns (cricsheet_name, was_resolved).

    Examples:
      "Virat Kohli"    -> ("V Kohli", True)
      "V Kohli"        -> ("V Kohli", True)       # already correct
      "Rohit Sharma"   -> ("RG Sharma", True)
      "Unknown Dude"   -> ("Unknown Dude", False)  # can't resolve
    """
    # 1. Already a known cricsheet name?
    if name in all_cricsheet_names:
        return name, True

    # 2. Direct lookup from full name mapping
    key = name.strip().lower()
    if key in full_to_cricsheet:
        return full_to_cricsheet[key], True

    # 3. Last-name match — find all names sharing the same last name
    #    then pick the best by first-name similarity
    last_name = name.strip().split()[-1].lower()
    first_name = name.strip().split()[0].lower() if len(name.strip().split()) > 1 else ""
    last_name_matches = [n for n in all_cricsheet_names if n.lower().split()[-1] == last_name]

    if last_name_matches:
        if len(last_name_matches) == 1:
            return last_name_matches[0], True
        # Multiple matches — pick by best overall similarity
        best_score, best_match = 0, last_name_matches[0]
        for candidate in last_name_matches:
            score = SequenceMatcher(None, key, candidate.lower()).ratio()
            # Boost if first initial matches
            if first_name and candidate[0].lower() == first_name[0]:
                score += 0.15
            if score > best_score:
                best_score, best_match = score, candidate
        return best_match, True

    # 4. General fuzzy match (fallback)
    best_score = 0
    best_match = None
    for known in all_cricsheet_names:
        score = SequenceMatcher(None, key, known.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = known

    if best_score >= 0.65:
        return best_match, True

    return name, False


def get_player_stats(name, ipl_stats, lifetime_stats):
    if name in ipl_stats:
        return ipl_stats[name]
    if name in lifetime_stats:
        return lifetime_stats[name]
    return DEFAULT_STATS.copy()


def infer_player_role(name, full_to_cricsheet, all_cricsheet_names, player_meta_lookup, ipl_stats, lifetime_stats):
    """Infer whether a player is primarily a batter, bowler, or all-rounder."""
    resolved_name, _ = resolve_player_name(name, full_to_cricsheet, all_cricsheet_names)
    stats = get_player_stats(resolved_name, ipl_stats, lifetime_stats)

    meta = player_meta_lookup.get(name.strip().lower()) or player_meta_lookup.get(resolved_name.strip().lower(), {})
    bowl_style = str(meta.get("bowl_style", "") or "").strip().lower()

    batting_innings = float(stats.get("batting_innings", 0) or 0)
    avg_runs = float(stats.get("avg_runs", 0) or 0)
    bowling_innings = float(stats.get("bowling_innings", 0) or 0)
    balls_bowled = float(stats.get("balls_bowled", 0) or 0)
    wickets = float(stats.get("wickets", 0) or 0)

    has_real_batting = batting_innings >= 15 and avg_runs >= 16
    has_real_bowling = wickets >= 12 or balls_bowled >= 360 or (bowling_innings >= 25 and wickets >= 8)

    if has_real_bowling and has_real_batting:
        return "all_rounder"
    if has_real_bowling:
        return "bowler"

    has_declared_bowling_style = bowl_style not in {"", "null", "none", "-", "na", "n/a"}
    if has_declared_bowling_style and (bowling_innings >= 8 or balls_bowled >= 96):
        return "all_rounder" if has_real_batting else "bowler"

    return "batter"


def format_player_option_label(
    player,
    overseas_set,
    full_to_cricsheet,
    all_cricsheet_names,
    player_meta_lookup,
    ipl_stats,
    lifetime_stats,
    recommended=False,
):
    """Build the dropdown label with role/overseas/recommendation markers."""
    role = infer_player_role(
        player,
        full_to_cricsheet,
        all_cricsheet_names,
        player_meta_lookup,
        ipl_stats,
        lifetime_stats,
    )

    parts = [player, ROLE_ICONS[role]]
    if player in overseas_set:
        parts.append("🌍")
    if recommended:
        parts.append("⭐")
    return "  ".join(parts)


def format_player_role_text(name, full_to_cricsheet, all_cricsheet_names, player_meta_lookup, ipl_stats, lifetime_stats):
    """Return a compact human-readable role label for a player."""
    role = infer_player_role(
        name,
        full_to_cricsheet,
        all_cricsheet_names,
        player_meta_lookup,
        ipl_stats,
        lifetime_stats,
    )
    return f"{ROLE_ICONS[role]} {ROLE_LABELS[role]}"


def render_account_sidebar():
    """Render login/register controls and return the active user."""
    current_user = st.session_state.get("auth_user")

    with st.sidebar:
        st.header("Account")
        st.caption("Predictions and Playing XI combinations are saved per account on this device.")

        if current_user:
            counts = get_user_dashboard_counts(current_user["id"])
            st.success(f"Logged in as @{current_user['username']}")
            st.caption(f"{counts['predictions']} predictions · {counts['combinations']} XI combinations")

            if st.button("Logout", key="logout_button", use_container_width=True):
                st.session_state["auth_user"] = None
                set_flash_message("info", "You have been logged out.")
                st.rerun()
        else:
            auth_mode = st.radio(
                "Access",
                ["Login", "Register"],
                key="auth_mode",
                horizontal=True,
            )

            if auth_mode == "Login":
                with st.form("login_form", clear_on_submit=False):
                    username = st.text_input("Username", key="login_username")
                    password = st.text_input("Password", type="password", key="login_password")
                    login_submitted = st.form_submit_button("Login", use_container_width=True)

                if login_submitted:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state["auth_user"] = user
                        set_flash_message("success", f"Welcome back, @{user['username']}.")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
            else:
                with st.form("register_form", clear_on_submit=True):
                    username = st.text_input("Username", key="register_username")
                    password = st.text_input("Password", type="password", key="register_password")
                    confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
                    register_submitted = st.form_submit_button("Create Account", use_container_width=True)

                if register_submitted:
                    validation_error = validate_new_user(username, password, confirm_password)
                    if validation_error:
                        st.error(validation_error)
                    else:
                        user, error_message = register_user(username, password)
                        if error_message:
                            st.error(error_message)
                        else:
                            st.session_state["auth_user"] = user
                            set_flash_message("success", f"Account created for @{user['username']}.")
                            st.rerun()

    return st.session_state.get("auth_user")


def render_user_dashboard(current_user):
    """Show saved predictions and XI combinations for the signed-in user."""
    st.subheader("My Dashboard")

    if not current_user:
        st.info("Login from the sidebar to save predictions, keep multiple Playing XI combinations, and access them later.")
        return

    counts = get_user_dashboard_counts(current_user["id"])
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Saved Predictions", counts["predictions"])
    metric_col2.metric("Saved XI Combinations", counts["combinations"])

    saved_combinations = list_user_xi_combinations(current_user["id"])
    prediction_history = get_user_prediction_history(current_user["id"], limit=15)

    st.markdown("#### Saved Playing XI Combinations")
    if saved_combinations:
        for team_name, team_code in TEAMS.items():
            team_combinations = [combo for combo in saved_combinations if combo["team_code"] == team_code]
            if not team_combinations:
                continue

            with st.expander(f"{team_name} · {len(team_combinations)} saved combination(s)", expanded=False):
                for combo in team_combinations:
                    impact_text = combo["impact_player"] or "None"
                    combo_info_col, combo_delete_col = st.columns([4.5, 1.1])
                    with combo_info_col:
                        st.markdown(f"**{combo['combo_name']}**")
                        st.caption(
                            f"Updated {format_saved_timestamp(combo['updated_at'])} · Impact player: {impact_text}"
                        )
                    with combo_delete_col:
                        st.write("")
                        if st.button(
                            "Delete",
                            key=f"delete_combo_dashboard_{combo['id']}",
                            use_container_width=True,
                        ):
                            if delete_xi_combination(current_user["id"], combo["id"]):
                                set_flash_message(
                                    "success",
                                    f"Deleted XI combination '{combo['combo_name']}'.",
                                )
                            else:
                                set_flash_message("warning", "Could not delete that XI combination.")
                            st.rerun()

                    st.write(", ".join(combo["playing_xi"]) if combo["playing_xi"] else "No players saved.")
                    st.markdown("---")
    else:
        st.info("No Playing XI combinations saved yet.")

    st.markdown("#### Recent Predictions")
    if prediction_history:
        for prediction in prediction_history:
            title = (
                f"{prediction['created_at']} · "
                f"{prediction['team1_code']} vs {prediction['team2_code']} · "
                f"{prediction['winner_code']} {prediction['winner_prob']:.1%}"
            )
            with st.expander(title, expanded=False):
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                summary_col1.metric("Winner", prediction["winner_name"])
                summary_col2.metric(prediction["team1_code"], f"{prediction['team1_prob']:.1%}")
                summary_col3.metric(prediction["team2_code"], f"{prediction['team2_prob']:.1%}")

                st.caption(
                    f"Saved {format_saved_timestamp(prediction['created_at'])} · Venue: {prediction['venue']} · Toss: {prediction['toss_winner']} chose to {prediction['toss_decision']}"
                )

                xi_col1, xi_col2 = st.columns(2)
                with xi_col1:
                    st.markdown(f"**{prediction['team1_name']} XI**")
                    st.write(", ".join(prediction["team1_xi"]) if prediction["team1_xi"] else "No XI saved.")
                    st.caption(f"Impact player: {prediction['team1_impact'] or 'None'}")
                with xi_col2:
                    st.markdown(f"**{prediction['team2_name']} XI**")
                    st.write(", ".join(prediction["team2_xi"]) if prediction["team2_xi"] else "No XI saved.")
                    st.caption(f"Impact player: {prediction['team2_impact'] or 'None'}")

                delete_prediction_col = st.columns([3.6, 1])[1]
                with delete_prediction_col:
                    if st.button(
                        "Delete Prediction",
                        key=f"delete_prediction_{prediction['id']}",
                        use_container_width=True,
                    ):
                        if delete_prediction_history_entry(current_user["id"], prediction["id"]):
                            set_flash_message("success", "Deleted saved prediction.")
                        else:
                            set_flash_message("warning", "Could not delete that saved prediction.")
                        st.rerun()
    else:
        st.info("No saved predictions yet. Run a prediction while logged in and it will appear here.")


def render_all_squads_browser():
    """Render all 2026 squads with the same role and overseas markers used in selectors."""
    st.subheader("All IPL 2026 Squads")
    st.caption("Role and overseas markers follow the same player metadata and stat heuristics used in the Playing XI selectors.")

    squad_cols = st.columns(2)
    for idx, (team_name, team_code) in enumerate(TEAMS.items()):
        squad = SQUADS_2026.get(team_code, [])
        overseas_set = OVERSEAS_PLAYERS.get(team_code, set())
        formatted_players = [
            format_player_option_label(
                player,
                overseas_set,
                full_to_cricsheet,
                all_cricsheet_names,
                player_meta_lookup,
                ipl_stats,
                lifetime_stats,
            )
            for player in squad
        ]
        escaped_players = "<br>".join(html.escape(player) for player in formatted_players)
        overseas_count = sum(1 for player in squad if player in overseas_set)

        with squad_cols[idx % 2]:
            logo_path = f"{LOGO_DIR}/{team_code}.png"
            logo_col, text_col = st.columns([0.28, 1.7])
            if os.path.exists(logo_path):
                logo_col.image(logo_path, width=50)
            text_col.markdown(f"**{team_name}**")
            text_col.caption(f"{team_code} · {len(squad)} players · {overseas_count} overseas")

            st.markdown(
                f"""
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 14px 16px; margin-bottom: 12px; line-height: 1.8;">
                    {escaped_players}
                </div>
                """,
                unsafe_allow_html=True,
            )


def format_saved_timestamp(timestamp):
    """Return a human-friendly label for saved record timestamps."""
    try:
        return datetime.fromisoformat(timestamp).strftime("%d %b %Y, %I:%M %p")
    except (TypeError, ValueError):
        return timestamp or "Unknown time"


def render_match_setup_controls(team_names, venues):
    """Render the top-of-page match controls and return the current selection."""
    st.markdown(
        """
        <div class="section-kicker">Prediction Workspace</div>
        <div class="page-section-title">Match Setup</div>
        <div class="page-section-subtitle">
            Choose the teams, venue, and toss setup here, then build the two Playing XIs below.
        </div>
        """,
        unsafe_allow_html=True,
    )

    setup_col1, setup_col2, setup_col3, setup_col4 = st.columns(4)

    with setup_col1:
        with st.container(border=True):
            st.markdown('<div class="match-chip-label">Team 1</div>', unsafe_allow_html=True)
            team1_name = st.selectbox(
                "Team 1",
                team_names,
                index=0,
                key="match_team1",
                label_visibility="collapsed",
            )

    with setup_col2:
        with st.container(border=True):
            st.markdown('<div class="match-chip-label">Team 2</div>', unsafe_allow_html=True)
            default_team2_index = 1 if len(team_names) > 1 else 0
            team2_name = st.selectbox(
                "Team 2",
                team_names,
                index=default_team2_index,
                key="match_team2",
                label_visibility="collapsed",
            )

    with setup_col3:
        with st.container(border=True):
            st.markdown('<div class="match-chip-label">Venue</div>', unsafe_allow_html=True)
            venue = st.selectbox(
                "Venue",
                venues,
                key="match_venue",
                label_visibility="collapsed",
            )

    with setup_col4:
        with st.container(border=True):
            st.markdown('<div class="match-chip-label">Toss</div>', unsafe_allow_html=True)
            toss_winner = st.selectbox(
                "Toss Winner",
                [team1_name, team2_name],
                key="match_toss_winner",
                label_visibility="collapsed",
            )
            toss_decision = st.radio(
                "Toss Decision",
                ["field", "bat"],
                key="match_toss_decision",
                horizontal=True,
                label_visibility="collapsed",
            )

    if team1_name == team2_name:
        st.warning("Please select two different teams.")
        st.stop()

    return team1_name, team2_name, venue, toss_winner, toss_decision


def render_selector_summary(selected_count, overseas_count, indian_count, strategy_text):
    """Render a compact summary row for a team selector."""
    overseas_value = f"{overseas_count}/4"
    if overseas_count > 4:
        overseas_value = f"{overseas_count}/4 max exceeded"

    st.markdown(
        f"""
        <div class="selector-stats">
            <div class="selector-stat">
                <div class="selector-stat-label">Playing XI</div>
                <div class="selector-stat-value">{selected_count}/11 selected</div>
            </div>
            <div class="selector-stat">
                <div class="selector-stat-label">Overseas Balance</div>
                <div class="selector-stat-value">🌍 {overseas_value} · 🇮🇳 {indian_count} Indian</div>
            </div>
            <div class="selector-stat">
                <div class="selector-stat-label">Impact Lens</div>
                <div class="selector-stat-value">{html.escape(strategy_text)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_team(name, alias_map):
    if pd.isna(name):
        return name
    return alias_map.get(name.strip().lower(), name.strip())


def get_team_form(team, match_history, n=5):
    team_matches = [m for m in match_history if m["team1"] == team or m["team2"] == team]
    recent = team_matches[-n:]
    if not recent:
        return 0.5
    return sum(1 for m in recent if m["winner"] == team) / len(recent)


def get_h2h(team1, team2, match_history):
    h2h = [
        m for m in match_history
        if (m["team1"] == team1 and m["team2"] == team2)
        or (m["team1"] == team2 and m["team2"] == team1)
    ]
    if not h2h:
        return 0.5, 0, 0
    t1_wins = sum(1 for m in h2h if m["winner"] == team1)
    t2_wins = len(h2h) - t1_wins
    return t1_wins / len(h2h), t1_wins, t2_wins


def get_venue_stats(team, venue, match_history):
    venue_matches = [m for m in match_history if m["venue"] == venue]

    team_venue = [m for m in venue_matches if m["team1"] == team or m["team2"] == team]
    team_venue_wr = sum(1 for m in team_venue if m["winner"] == team) / len(team_venue) if team_venue else 0.5

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
    venue_l = venue.lower()
    if home and home.lower() in venue_l:
        return 1
    if secondary and secondary.lower() in venue_l:
        return 1
    return 0


def team_bats_first(team_code, toss_winner_code, toss_decision):
    """Return True if this team is projected to bat first."""
    if toss_winner_code == team_code:
        return toss_decision == "bat"
    return toss_decision == "field"


def summarize_lineup_stats(stats_list):
    """Collapse a list of player stats into model-ready lineup averages."""
    return {
        "batting_sr": float(np.mean([s["batting_sr"] for s in stats_list])),
        "bowling_econ": float(np.mean([s["bowling_econ"] for s in stats_list])),
        "experience": float(np.mean([s["experience"] for s in stats_list])),
        "avg_runs": float(np.mean([s["avg_runs"] for s in stats_list])),
    }


def batting_score(stats):
    """Simple batting composite used to judge impact-batter upgrades."""
    return float(stats["avg_runs"]) + (float(stats["batting_sr"]) / 10.0)


def bowling_score(stats):
    """
    Bowling composite used to judge impact-bowler upgrades.
    Non-bowlers are intentionally penalized so batting-only players are the
    most likely outgoing option when a team brings in a specialist bowler.
    """
    bowling_innings = float(stats.get("bowling_innings", 0) or 0)
    balls_bowled = float(stats.get("balls_bowled", 0) or 0)
    if bowling_innings <= 0 and balls_bowled <= 0:
        return -5.0

    econ = float(stats["bowling_econ"]) if stats["bowling_econ"] > 0 else 9.5
    sample_bonus = min(balls_bowled / 120.0, 1.0) + min(bowling_innings / 10.0, 1.0) * 0.25
    return (10.5 - econ) + sample_bonus


def apply_impact_substitution(team_xi, impact_player, phase, ipl_stats, lifetime_stats):
    """
    Model the most likely one-for-one impact swap for either batting or bowling.
    The model features stay unchanged; we only update the effective lineup stats
    for the innings where the impact substitute is most likely to matter.
    """
    base_names = list(team_xi)
    base_stats = [get_player_stats(p, ipl_stats, lifetime_stats) for p in base_names]
    baseline = summarize_lineup_stats(base_stats)

    phase_result = {
        "stats": baseline,
        "used": False,
        "impact_player": impact_player,
        "replaced_player": None,
        "score_gain": 0.0,
        "phase": phase,
    }

    if not impact_player or len(base_names) != 11:
        return phase_result

    impact_stats = get_player_stats(impact_player, ipl_stats, lifetime_stats)

    if phase == "batting":
        scores = [batting_score(s) for s in base_stats]
        candidate_score = batting_score(impact_stats)
    else:
        scores = [bowling_score(s) for s in base_stats]
        candidate_score = bowling_score(impact_stats)

    replace_idx = int(np.argmin(scores))
    replaced_score = scores[replace_idx]

    if candidate_score <= replaced_score:
        return phase_result

    phase_stats = list(base_stats)
    phase_stats[replace_idx] = impact_stats

    phase_result.update({
        "stats": summarize_lineup_stats(phase_stats),
        "used": True,
        "replaced_player": base_names[replace_idx],
        "score_gain": float(candidate_score - replaced_score),
    })
    return phase_result


def build_effective_team_strength(team_xi, impact_player, bats_first, ipl_stats, lifetime_stats):
    """
    IPL impact player rule approximation:
    - team batting first is more likely to bring in a bowler while defending
    - team chasing is more likely to bring in a batter for the chase
    """
    base_stats = [get_player_stats(p, ipl_stats, lifetime_stats) for p in team_xi]
    baseline = summarize_lineup_stats(base_stats)

    batting_phase = {
        "stats": baseline,
        "used": False,
        "impact_player": impact_player,
        "replaced_player": None,
        "score_gain": 0.0,
        "phase": "batting",
    }
    bowling_phase = {
        "stats": baseline,
        "used": False,
        "impact_player": impact_player,
        "replaced_player": None,
        "score_gain": 0.0,
        "phase": "bowling",
    }

    primary_phase = "bowling" if bats_first else "batting"
    primary_result = apply_impact_substitution(team_xi, impact_player, primary_phase, ipl_stats, lifetime_stats)
    if primary_phase == "batting":
        batting_phase = primary_result
    else:
        bowling_phase = primary_result

    effective = {
        "batting_sr": batting_phase["stats"]["batting_sr"],
        "bowling_econ": bowling_phase["stats"]["bowling_econ"],
        "experience": float(np.mean([
            batting_phase["stats"]["experience"],
            bowling_phase["stats"]["experience"],
        ])),
        "avg_runs": batting_phase["stats"]["avg_runs"],
    }

    profile = {
        "impact_player": impact_player,
        "used": batting_phase["used"] or bowling_phase["used"],
        "strategy": "bowling" if bats_first else "batting",
        "baseline": baseline,
        "effective": effective,
        "batting_phase": batting_phase,
        "bowling_phase": bowling_phase,
    }
    return effective, profile


def recommend_impact_player(
    team_xi,
    squad,
    overseas_set,
    bats_first,
    ipl_stats,
    lifetime_stats,
    full_to_cricsheet,
    all_cricsheet_names,
):
    """Pick the bench player with the biggest expected impact gain."""
    if len(team_xi) != 11 or not squad:
        return None

    overseas_in_xi = sum(1 for p in team_xi if p in overseas_set)
    resolved_xi = [
        resolve_player_name(player, full_to_cricsheet, all_cricsheet_names)[0]
        for player in team_xi
    ]

    best_player = None
    best_gain = 0.0

    for player in squad:
        if player in team_xi:
            continue
        if player in overseas_set and overseas_in_xi >= 4:
            continue

        resolved_player = resolve_player_name(player, full_to_cricsheet, all_cricsheet_names)[0]
        _, profile = build_effective_team_strength(
            resolved_xi,
            resolved_player,
            bats_first,
            ipl_stats,
            lifetime_stats,
        )

        phase_key = "bowling_phase" if bats_first else "batting_phase"
        gain = profile[phase_key]["score_gain"]
        if gain > best_gain:
            best_player = player
            best_gain = gain

    return best_player


def predict_match(
    team1_code, team2_code, venue, toss_winner_code, toss_decision,
    team1_xi, team2_xi, model, feature_cols, elo_ratings, match_history,
    ipl_stats, lifetime_stats,
    team1_impact_player=None, team2_impact_player=None,
):
    """Run prediction and return results dict."""
    # Elo
    t1_elo = elo_ratings.get(team1_code, 1500)
    t2_elo = elo_ratings.get(team2_code, 1500)

    # Form
    t1_form = get_team_form(team1_code, match_history)
    t2_form = get_team_form(team2_code, match_history)

    # H2H
    h2h_wr, t1_h2h_wins, t2_h2h_wins = get_h2h(team1_code, team2_code, match_history)

    # Venue
    t1_venue_wr, bat_first_pct = get_venue_stats(team1_code, venue, match_history)
    t2_venue_wr, _ = get_venue_stats(team2_code, venue, match_history)

    # Toss
    toss_is_t1 = 1 if toss_winner_code == team1_code else 0
    chose_bat = 1 if toss_decision == "bat" else 0

    # Home
    t1_home = is_home(team1_code, venue)
    t2_home = is_home(team2_code, venue)

    # Player strength, adjusted for the projected impact substitute.
    t1_bats_first = team_bats_first(team1_code, toss_winner_code, toss_decision)
    t2_bats_first = team_bats_first(team2_code, toss_winner_code, toss_decision)

    t1_strength, t1_impact_profile = build_effective_team_strength(
        team1_xi,
        team1_impact_player,
        t1_bats_first,
        ipl_stats,
        lifetime_stats,
    )
    t2_strength, t2_impact_profile = build_effective_team_strength(
        team2_xi,
        team2_impact_player,
        t2_bats_first,
        ipl_stats,
        lifetime_stats,
    )

    t1_bat_sr = t1_strength["batting_sr"]
    t1_bowl_econ = t1_strength["bowling_econ"]
    t1_exp = t1_strength["experience"]
    t1_avg_runs = t1_strength["avg_runs"]

    t2_bat_sr = t2_strength["batting_sr"]
    t2_bowl_econ = t2_strength["bowling_econ"]
    t2_exp = t2_strength["experience"]
    t2_avg_runs = t2_strength["avg_runs"]

    features = pd.DataFrame([{
        "team1_elo": t1_elo,
        "team2_elo": t2_elo,
        "elo_diff": t1_elo - t2_elo,
        "team1_form": t1_form,
        "team2_form": t2_form,
        "form_diff": t1_form - t2_form,
        "h2h_team1_wr": h2h_wr,
        "team1_venue_wr": t1_venue_wr,
        "team2_venue_wr": t2_venue_wr,
        "bat_first_pct": bat_first_pct,
        "toss_winner_is_team1": toss_is_t1,
        "toss_chose_bat": chose_bat,
        "team1_home": t1_home,
        "team2_home": t2_home,
        "team1_bat_sr": t1_bat_sr,
        "team1_bowl_econ": t1_bowl_econ,
        "team1_experience": t1_exp,
        "team1_avg_runs": t1_avg_runs,
        "team2_bat_sr": t2_bat_sr,
        "team2_bowl_econ": t2_bowl_econ,
        "team2_experience": t2_exp,
        "team2_avg_runs": t2_avg_runs,
        "bat_sr_diff": t1_bat_sr - t2_bat_sr,
        "bowl_econ_diff": t1_bowl_econ - t2_bowl_econ,
        "experience_diff": t1_exp - t2_exp,
    }])

    prob = model.predict_proba(features[feature_cols])[0]
    t1_prob = prob[1]
    t2_prob = prob[0]

    return {
        "t1_prob": t1_prob,
        "t2_prob": t2_prob,
        "t1_elo": t1_elo,
        "t2_elo": t2_elo,
        "t1_form": t1_form,
        "t2_form": t2_form,
        "h2h_t1_wins": t1_h2h_wins,
        "h2h_t2_wins": t2_h2h_wins,
        "t1_venue_wr": t1_venue_wr,
        "t2_venue_wr": t2_venue_wr,
        "bat_first_pct": bat_first_pct,
        "t1_home": t1_home,
        "t2_home": t2_home,
        "t1_bat_sr": t1_bat_sr,
        "t2_bat_sr": t2_bat_sr,
        "t1_bowl_econ": t1_bowl_econ,
        "t2_bowl_econ": t2_bowl_econ,
        "t1_exp": t1_exp,
        "t2_exp": t2_exp,
        "team1_impact": t1_impact_profile,
        "team2_impact": t2_impact_profile,
    }


# ============================================================
# Load everything
# ============================================================
init_app_database()
ensure_app_session_state()

if not os.path.exists("Models/xgb_model.pkl"):
    st.error("Model not found! Run `training.ipynb` first to train and save the model.")
    st.stop()

model, feature_cols, elo_ratings, alias_map, match_history = load_model()
ipl_stats, lifetime_stats = load_player_stats()
full_to_cricsheet, players_meta = build_name_resolver()
player_meta_lookup = build_player_meta_lookup(players_meta)

# All known cricsheet-format names (for fuzzy matching)
all_cricsheet_names = sorted(set(list(ipl_stats.keys()) + list(lifetime_stats.keys())))

# Load IPL 2026 squads for player selection
SQUADS_2026 = {}
squads_path = "Datasets/ipl_2026_squads.json"
if os.path.exists(squads_path):
    with open(squads_path) as f:
        _squads_data = json.load(f)
    for code, data in _squads_data["teams"].items():
        SQUADS_2026[code] = data["squad"]


# ============================================================
# UI — Header
# ============================================================
st.markdown('<p class="main-title">IPL 2026 Match Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">XGBoost ML Model trained on 18 seasons of IPL data (2008-2025)</p>', unsafe_allow_html=True)
render_flash_message()
current_user = render_account_sidebar()
st.markdown("---")

# ============================================================
# UI — Match Setup
# ============================================================
team_names = list(TEAMS.keys())
team1_name, team2_name, venue, toss_winner, toss_decision = render_match_setup_controls(team_names, VENUES)
st.caption("Select the Playing XI and projected impact player for each team.")

# ============================================================
# UI — Playing XI Input
# ============================================================
team1_code = TEAMS[team1_name]
team2_code = TEAMS[team2_name]
toss_winner_code = TEAMS[toss_winner]
team1_bats_first = team_bats_first(team1_code, toss_winner_code, toss_decision)
team2_bats_first = team_bats_first(team2_code, toss_winner_code, toss_decision)

col_xi1, col_xi2 = st.columns(2)

# Get squad lists for selected teams
squad1 = SQUADS_2026.get(team1_code, [])
squad2 = SQUADS_2026.get(team2_code, [])
overseas1 = OVERSEAS_PLAYERS.get(team1_code, set())
overseas2 = OVERSEAS_PLAYERS.get(team2_code, set())


def render_xi_selector(col, team_name, team_code, squad, overseas_set, bats_first, key_suffix, current_user):
    """Render the Playing XI + impact player selector with user-saved combinations."""
    with col:
        logo_path = f"{LOGO_DIR}/{team_code}.png"
        c_logo, c_name = st.columns([0.3, 2])
        if os.path.exists(logo_path):
            c_logo.image(logo_path, width=55)
        c_name.subheader(team_name)
        st.caption("Role icons: 🏏 batter · 🎯 bowler · 🧰 all-rounder · 🌍 overseas")

        if squad:
            player_to_display = {
                player: format_player_option_label(
                    player,
                    overseas_set,
                    full_to_cricsheet,
                    all_cricsheet_names,
                    player_meta_lookup,
                    ipl_stats,
                    lifetime_stats,
                )
                for player in squad
            }
            display_options = list(player_to_display.values())
            option_map = {label: player for player, label in player_to_display.items()}

            xi_state_key = f"xi_{key_suffix}_select"
            xi_team_state_key = f"xi_{key_suffix}_team"
            impact_state_key = f"impact_{key_suffix}_select"
            combo_select_key = f"combo_{key_suffix}_selected"
            combo_name_key = f"combo_name_{key_suffix}"

            saved_combinations = []
            combo_lookup = {}
            default_combo = None

            if current_user:
                raw_combinations = get_user_xi_combinations_for_team(current_user["id"], team_code, squad)
                for combo in raw_combinations:
                    combo_display = [
                        player_to_display[player]
                        for player in combo["filtered_xi"]
                        if player in player_to_display
                    ]
                    impact_display = "None"
                    if combo["valid_impact_player"]:
                        impact_display = format_player_option_label(
                            combo["valid_impact_player"],
                            overseas_set,
                            full_to_cricsheet,
                            all_cricsheet_names,
                            player_meta_lookup,
                            ipl_stats,
                            lifetime_stats,
                        )

                    label = combo["combo_name"]
                    if combo["missing_players"]:
                        label = f"{combo['combo_name']} · Needs update"

                    saved_combinations.append(
                        {
                            **combo,
                            "label": label,
                            "display_xi": combo_display,
                            "impact_display": impact_display,
                            "updated_display": format_saved_timestamp(combo["updated_at"]),
                        }
                    )

                combo_lookup = {combo["label"]: combo for combo in saved_combinations}
                default_combo = saved_combinations[0] if saved_combinations else None

            if st.session_state.get(xi_team_state_key) != team_code:
                st.session_state[xi_team_state_key] = team_code
                if default_combo:
                    st.session_state[xi_state_key] = list(default_combo["display_xi"])
                    st.session_state[impact_state_key] = default_combo["impact_display"]
                    st.session_state[combo_select_key] = default_combo["label"]
                    st.session_state[combo_name_key] = default_combo["combo_name"]
                else:
                    st.session_state[xi_state_key] = []
                    st.session_state[impact_state_key] = "None"
                    st.session_state[combo_name_key] = f"{team_code} XI 1"
                    st.session_state.pop(combo_select_key, None)

            st.session_state.setdefault(combo_name_key, default_combo["combo_name"] if default_combo else f"{team_code} XI 1")

            if current_user:
                if saved_combinations:
                    with st.expander(f"Saved XI combinations ({len(saved_combinations)})", expanded=False):
                        combo_col, load_col, delete_col = st.columns([1.45, 0.75, 0.8])
                        with combo_col:
                            selected_combo_label = st.selectbox(
                                f"Saved combinations ({team_code})",
                                options=[combo["label"] for combo in saved_combinations],
                                key=combo_select_key,
                                label_visibility="collapsed",
                            )
                        selected_combo = combo_lookup[selected_combo_label]
                        with load_col:
                            st.write("")
                            st.write("")
                            st.button(
                                "Load",
                                key=f"load_xi_{key_suffix}",
                                use_container_width=True,
                                on_click=load_combination_into_state,
                                args=(
                                    xi_state_key,
                                    list(selected_combo["display_xi"]),
                                    impact_state_key,
                                    selected_combo["impact_display"],
                                    combo_name_key,
                                    selected_combo["combo_name"],
                                ),
                            )
                        with delete_col:
                            st.write("")
                            st.write("")
                            if st.button(
                                "Delete",
                                key=f"delete_combo_selector_{selected_combo['id']}",
                                use_container_width=True,
                            ):
                                if delete_xi_combination(current_user["id"], selected_combo["id"]):
                                    set_flash_message(
                                        "success",
                                        f"Deleted XI combination '{selected_combo['combo_name']}'.",
                                    )
                                else:
                                    set_flash_message("warning", "Could not delete that XI combination.")
                                st.rerun()

                        st.caption(
                            f"Updated {selected_combo['updated_display']} · Impact player: {selected_combo['impact_player'] or 'None'}"
                        )
                        if selected_combo["missing_players"]:
                            st.caption(
                                "Needs update: "
                                + ", ".join(selected_combo["missing_players"])
                                + " not found in the current squad."
                            )
                else:
                    st.caption("No saved XI combinations yet. Save one below once this team setup is ready.")
            else:
                st.caption("Login to save and reuse multiple XI combinations.")

            st.markdown("##### Playing XI")
            selected_display = st.multiselect(
                f"Select Playing XI ({team_code})",
                options=display_options,
                max_selections=11,
                key=xi_state_key,
                label_visibility="collapsed",
            )
            selected_real = [option_map[label] for label in selected_display if label in option_map]

            overseas_count = sum(1 for player in selected_real if player in overseas_set)
            indian_count = len(selected_real) - overseas_count

            if overseas_count > 4:
                st.error(f"Too many overseas players! You selected {overseas_count} — maximum is 4.")

            strategy_text = "bowling impact while defending" if bats_first else "batting impact while chasing"
            render_selector_summary(len(selected_real), overseas_count, indian_count, strategy_text)

            if len(selected_real) != 11:
                st.session_state[impact_state_key] = "Complete the Playing XI first"
                st.markdown("##### Impact Player")
                st.selectbox(
                    f"Projected Impact Player ({team_code})",
                    options=["Complete the Playing XI first"],
                    index=0,
                    disabled=True,
                    key=impact_state_key,
                    label_visibility="collapsed",
                )
                if current_user:
                    st.caption("Complete all 11 players to save or update a named combination.")
                return selected_real, overseas_count, None

            recommended_player = recommend_impact_player(
                selected_real,
                squad,
                overseas_set,
                bats_first,
                ipl_stats,
                lifetime_stats,
                full_to_cricsheet,
                all_cricsheet_names,
            )

            bench_players = [player for player in squad if player not in selected_real]
            impact_options = ["None"]
            impact_option_map = {"None": None}

            for player in bench_players:
                label = format_player_option_label(
                    player,
                    overseas_set,
                    full_to_cricsheet,
                    all_cricsheet_names,
                    player_meta_lookup,
                    ipl_stats,
                    lifetime_stats,
                    recommended=(player == recommended_player),
                )
                impact_options.append(label)
                impact_option_map[label] = player

            default_label = "None"
            if recommended_player:
                default_label = next(
                    (label for label, player in impact_option_map.items() if player == recommended_player),
                    "None",
                )

            current_impact_value = st.session_state.get(impact_state_key, default_label)
            if current_impact_value not in impact_options:
                st.session_state[impact_state_key] = default_label
                current_impact_value = default_label

            st.markdown("##### Impact Player")
            selected_impact_display = st.selectbox(
                f"Projected Impact Player ({team_code})",
                options=impact_options,
                index=impact_options.index(current_impact_value),
                key=impact_state_key,
                help="One projected substitute from the bench. The app models this player as a batting boost while chasing or a bowling boost while defending.",
                label_visibility="collapsed",
            )
            selected_impact = impact_option_map[selected_impact_display]

            if selected_impact and selected_impact in overseas_set and overseas_count >= 4:
                st.error("Overseas impact player not allowed when the starting XI already has 4 overseas players.")
            elif selected_impact:
                role_text = format_player_role_text(
                    selected_impact,
                    full_to_cricsheet,
                    all_cricsheet_names,
                    player_meta_lookup,
                    ipl_stats,
                    lifetime_stats,
                )
                st.caption(f"Projected impact player: {selected_impact} ({role_text}). Model context: {strategy_text}.")

            if current_user:
                st.markdown("##### Save or Update Combination")
                save_name_col, save_button_col = st.columns([1.6, 1])
                with save_name_col:
                    combo_name = st.text_input(
                        f"Combination name ({team_code})",
                        key=combo_name_key,
                        placeholder=f"{team_code} XI 1",
                        help="Use different names to keep multiple Playing XI combinations for the same team.",
                        label_visibility="collapsed",
                    )
                with save_button_col:
                    st.write("")
                    st.write("")
                    if st.button(
                        f"Save Combination ({team_code})",
                        key=f"save_xi_{key_suffix}",
                        use_container_width=True,
                        disabled=not combo_name.strip(),
                    ):
                        action = save_xi_combination(
                            current_user["id"],
                            team_code,
                            team_name,
                            combo_name,
                            selected_real,
                            selected_impact,
                        )
                        set_flash_message(
                            "success",
                            f"{team_code} combination '{combo_name.strip()}' {action}.",
                        )
                        st.rerun()
            else:
                st.caption("Login to save this XI and projected impact-player combination.")

            return selected_real, overseas_count, selected_impact
        else:
            xi_text = st.text_area(
                f"Playing XI — {team_code}",
                height=280,
                placeholder="Player 1\nPlayer 2\n...",
                label_visibility="collapsed",
            )
            xi = [name.strip() for name in xi_text.strip().split("\n") if name.strip()]
            impact_text = st.text_input(
                f"Projected Impact Player ({team_code})",
                key=f"impact_{key_suffix}_text",
                help="Optional when squad auto-fill is unavailable.",
            ).strip()
            return xi, 0, impact_text or None


team1_xi, t1_overseas_count, team1_impact = render_xi_selector(
    col_xi1, team1_name, team1_code, squad1, overseas1, team1_bats_first, "1", current_user
)
team2_xi, t2_overseas_count, team2_impact = render_xi_selector(
    col_xi2, team2_name, team2_code, squad2, overseas2, team2_bats_first, "2", current_user
)

# ============================================================
# UI — Predict Button
# ============================================================
st.markdown("")
col_btn_left, col_btn, col_btn_right = st.columns([2, 1, 2])
with col_btn:
    predict_clicked = st.button("Predict Winner", use_container_width=True, type="primary")

# ============================================================
# Validation & Prediction
# ============================================================
if predict_clicked:
    # Validate XI
    errors = []
    if len(team1_xi) == 0:
        errors.append(f"Enter Playing XI for {team1_name}")
    elif len(team1_xi) != 11:
        errors.append(f"{team1_name}: selected {len(team1_xi)} players (need exactly 11)")
    if t1_overseas_count > 4:
        errors.append(f"{team1_name}: {t1_overseas_count} overseas players selected (max 4 allowed)")
    if team1_impact and team1_impact in overseas1 and t1_overseas_count >= 4:
        errors.append(f"{team1_name}: overseas impact player not allowed when the starting XI already has 4 overseas players")
    if len(team2_xi) == 0:
        errors.append(f"Enter Playing XI for {team2_name}")
    elif len(team2_xi) != 11:
        errors.append(f"{team2_name}: selected {len(team2_xi)} players (need exactly 11)")
    if t2_overseas_count > 4:
        errors.append(f"{team2_name}: {t2_overseas_count} overseas players selected (max 4 allowed)")
    if team2_impact and team2_impact in overseas2 and t2_overseas_count >= 4:
        errors.append(f"{team2_name}: overseas impact player not allowed when the starting XI already has 4 overseas players")

    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    entered_team1_xi = list(team1_xi)
    entered_team2_xi = list(team2_xi)
    entered_team1_impact = team1_impact
    entered_team2_impact = team2_impact

    # --- Resolve player names ---
    # Convert full names (e.g. "Virat Kohli") to cricsheet format ("V Kohli")
    resolved_t1 = []
    resolved_t2 = []
    resolved_names_t1 = []  # for display
    resolved_names_t2 = []
    unknown_t1 = []
    unknown_t2 = []
    impact_resolutions = []

    for name in team1_xi:
        resolved, found = resolve_player_name(name, full_to_cricsheet, all_cricsheet_names)
        resolved_t1.append(resolved)
        if not found:
            unknown_t1.append(name)
        elif resolved != name:
            resolved_names_t1.append(f"{name} -> **{resolved}**")

    for name in team2_xi:
        resolved, found = resolve_player_name(name, full_to_cricsheet, all_cricsheet_names)
        resolved_t2.append(resolved)
        if not found:
            unknown_t2.append(name)
        elif resolved != name:
            resolved_names_t2.append(f"{name} -> **{resolved}**")

    resolved_t1_impact = None
    if team1_impact:
        resolved_t1_impact, found = resolve_player_name(team1_impact, full_to_cricsheet, all_cricsheet_names)
        if not found:
            unknown_t1.append(team1_impact)
        elif resolved_t1_impact != team1_impact:
            impact_resolutions.append(f"**{team1_code}** impact: {team1_impact} -> **{resolved_t1_impact}**")

    resolved_t2_impact = None
    if team2_impact:
        resolved_t2_impact, found = resolve_player_name(team2_impact, full_to_cricsheet, all_cricsheet_names)
        if not found:
            unknown_t2.append(team2_impact)
        elif resolved_t2_impact != team2_impact:
            impact_resolutions.append(f"**{team2_code}** impact: {team2_impact} -> **{resolved_t2_impact}**")

    # Show name resolutions
    if resolved_names_t1:
        st.info(f"**{team1_code}** name matches: " + ", ".join(resolved_names_t1))
    if resolved_names_t2:
        st.info(f"**{team2_code}** name matches: " + ", ".join(resolved_names_t2))
    if impact_resolutions:
        st.info(" | ".join(impact_resolutions))

    if unknown_t1:
        st.warning(f"**{team1_code}** — unknown players (using defaults): {', '.join(unknown_t1)}")
    if unknown_t2:
        st.warning(f"**{team2_code}** — unknown players (using defaults): {', '.join(unknown_t2)}")

    # Use resolved names for prediction
    team1_xi = resolved_t1
    team2_xi = resolved_t2

    # Run prediction
    toss_winner_code = TEAMS[toss_winner]
    result = predict_match(
        team1_code, team2_code, venue, toss_winner_code, toss_decision,
        team1_xi, team2_xi, model, feature_cols, elo_ratings, match_history,
        ipl_stats, lifetime_stats,
        team1_impact_player=resolved_t1_impact,
        team2_impact_player=resolved_t2_impact,
    )

    t1_prob = result["t1_prob"]
    t2_prob = result["t2_prob"]
    winner_name = team1_name if t1_prob > 0.5 else team2_name
    winner_code = team1_code if t1_prob > 0.5 else team2_code
    winner_prob = max(t1_prob, t2_prob)

    if current_user:
        save_prediction_history(
            current_user["id"],
            {
                "team1_name": team1_name,
                "team1_code": team1_code,
                "team2_name": team2_name,
                "team2_code": team2_code,
                "venue": venue,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "team1_xi": entered_team1_xi,
                "team2_xi": entered_team2_xi,
                "team1_impact": entered_team1_impact,
                "team2_impact": entered_team2_impact,
                "winner_name": winner_name,
                "winner_code": winner_code,
                "winner_prob": winner_prob,
                "team1_prob": t1_prob,
                "team2_prob": t2_prob,
            },
        )

    st.markdown("---")
    if current_user:
        st.success("Prediction saved to your account history.")

    # --- Winner Announcement ---
    st.markdown(f"""
    <div class="winner-box">
        <div class="winner-label">Predicted Winner</div>
        <div class="winner-name">{winner_name}</div>
        <div style="color: #b7e4c7; font-size: 1.2rem; margin-top: 8px;">
            {winner_prob:.1%} confidence
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Probability Bar ---
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; margin: 10px 0;">
        <span style="width: 60px; text-align: right; font-weight: 600;">{team1_code}</span>
        <div class="prob-bar" style="flex: 1;">
            <div class="prob-t1" style="width: {t1_prob*100:.1f}%;">{t1_prob:.1%}</div>
            <div class="prob-t2" style="width: {t2_prob*100:.1f}%;">{t2_prob:.1%}</div>
        </div>
        <span style="width: 60px; font-weight: 600;">{team2_code}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Impact Player Assumptions")
    st.caption("The app projects one bench substitute per team. Batting-first teams get a bowling-impact lens; chasing teams get a batting-impact lens.")

    def render_impact_card(col, team_name, team_code, profile):
        with col:
            phase_key = "bowling_phase" if profile["strategy"] == "bowling" else "batting_phase"
            phase_label = "Bowling swap while defending" if profile["strategy"] == "bowling" else "Batting swap while chasing"
            impact_name = profile["impact_player"] or "No impact player selected"
            role_text = None
            swap_html = ""
            if profile["impact_player"]:
                role_text = format_player_role_text(
                    profile["impact_player"],
                    full_to_cricsheet,
                    all_cricsheet_names,
                    player_meta_lookup,
                    ipl_stats,
                    lifetime_stats,
                )

            if profile["used"]:
                phase_profile = profile[phase_key]
                swap_html = dedent(f"""
                <div style="margin-top: 10px; padding: 10px 12px; background: #0d1117; border: 1px solid #30363d; border-radius: 8px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.9px; color: #8b949e; margin-bottom: 6px;">
                        Projected Substitution
                    </div>
                    <div style="font-size: 0.92rem; color: #e6edf3; font-weight: 600;">
                        <span style="color: #3fb950;">IN:</span> {impact_name}
                        <span style="color: #8b949e; margin: 0 10px;">|</span>
                        <span style="color: #f85149;">OUT:</span> {phase_profile['replaced_player']}
                    </div>
                </div>
                """).strip()
                if profile["strategy"] == "bowling":
                    delta_text = (
                        f"Bowling economy {profile['baseline']['bowling_econ']:.2f} -> "
                        f"{profile['effective']['bowling_econ']:.2f}"
                    )
                else:
                    delta_text = (
                        f"Bat SR {profile['baseline']['batting_sr']:.1f} -> {profile['effective']['batting_sr']:.1f} "
                        f"and Avg Runs {profile['baseline']['avg_runs']:.1f} -> {profile['effective']['avg_runs']:.1f}"
                    )
                body_text = (
                    f"Selected player: {impact_name}"
                    f"{f' ({role_text})' if role_text else ''}. "
                    f"Projected in the {phase_label.lower()} scenario. {delta_text}."
                )
                tone_color = "#58a6ff"
            elif profile["impact_player"]:
                swap_html = dedent("""
                <div style="margin-top: 10px; padding: 10px 12px; background: #0d1117; border: 1px solid #30363d; border-radius: 8px;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.9px; color: #8b949e; margin-bottom: 6px;">
                        Projected Substitution
                    </div>
                    <div style="font-size: 0.9rem; color: #8b949e;">
                        No swap projected. Baseline XI retained.
                    </div>
                </div>
                """).strip()
                body_text = (
                    f"Selected player: {impact_name}"
                    f"{f' ({role_text})' if role_text else ''}. "
                    f"The current XI still projects stronger for "
                    f"{phase_label.lower()}, so the baseline XI was kept."
                )
                tone_color = "#f7c948"
            else:
                body_text = "No impact player selected. Prediction uses the Playing XI as entered."
                tone_color = "#8b949e"

            st.markdown(f"""
            <div style="background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 16px; margin: 8px 0;">
                <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #8b949e;">
                    {team_name} Impact Plan
                </div>
                <div style="font-size: 1.05rem; font-weight: 700; color: {tone_color}; margin-top: 4px;">
                    {phase_label}
                </div>
                <div style="font-size: 0.85rem; color: #e6edf3; margin-top: 8px;">
                    {body_text}
                </div>
                {swap_html}
            </div>
            """, unsafe_allow_html=True)

    impact_col1, impact_col2 = st.columns(2)
    render_impact_card(impact_col1, team1_name, team1_code, result["team1_impact"])
    render_impact_card(impact_col2, team2_name, team2_code, result["team2_impact"])

    # --- Visual Charts ---
    chart_col1, chart_col2 = st.columns(2)

    # ---- Chart 1: Radar Chart ----
    with chart_col1:
        # Normalize values to 0-100 scale for radar
        def norm(val, low, high):
            return max(0, min(100, (val - low) / (high - low) * 100))

        categories = ['Strength', 'Form', 'Head-to-Head', 'Batting', 'Bowling', 'Experience']
        t1_vals = [
            norm(result['t1_elo'], 1350, 1650),
            result['t1_form'] * 100,
            (result['h2h_t1_wins'] / max(result['h2h_t1_wins'] + result['h2h_t2_wins'], 1)) * 100,
            norm(result['t1_bat_sr'], 100, 160),
            norm(12 - result['t1_bowl_econ'], 0, 6) * 100 / 100,   # invert: lower econ = higher score
            norm(result['t1_exp'], 0, 200),
        ]
        t2_vals = [
            norm(result['t2_elo'], 1350, 1650),
            result['t2_form'] * 100,
            (result['h2h_t2_wins'] / max(result['h2h_t1_wins'] + result['h2h_t2_wins'], 1)) * 100,
            norm(result['t2_bat_sr'], 100, 160),
            norm(12 - result['t2_bowl_econ'], 0, 6) * 100 / 100,
            norm(result['t2_exp'], 0, 200),
        ]

        # Close the radar polygon
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        t1_vals_plot = t1_vals + [t1_vals[0]]
        t2_vals_plot = t2_vals + [t2_vals[0]]
        angles += [angles[0]]

        fig_radar, ax_radar = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        fig_radar.patch.set_facecolor('#0d1117')
        ax_radar.set_facecolor('#0d1117')

        ax_radar.plot(angles, t1_vals_plot, 'o-', linewidth=2, color='#2196F3', label=team1_code, markersize=5)
        ax_radar.fill(angles, t1_vals_plot, alpha=0.15, color='#2196F3')
        ax_radar.plot(angles, t2_vals_plot, 'o-', linewidth=2, color='#FF5722', label=team2_code, markersize=5)
        ax_radar.fill(angles, t2_vals_plot, alpha=0.15, color='#FF5722')

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, size=8, color='#8b949e')
        ax_radar.set_ylim(0, 100)
        ax_radar.set_yticks([25, 50, 75])
        ax_radar.set_yticklabels([])
        ax_radar.spines['polar'].set_color('#30363d')
        ax_radar.grid(color='#30363d', linewidth=0.5)
        ax_radar.tick_params(colors='#8b949e')

        leg = ax_radar.legend(loc='upper right', bbox_to_anchor=(1.25, 1.12), fontsize=9,
                              facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')

        st.pyplot(fig_radar, use_container_width=True)
        plt.close(fig_radar)
        st.caption("*Team comparison across 6 key dimensions — larger area = stronger.*")

    # ---- Chart 2: Factor Scorecard (matplotlib) ----
    with chart_col2:
        def factor_winner(t1_val, t2_val, higher_is_better=True, threshold=0.02):
            diff = abs(t1_val - t2_val)
            max_val = max(abs(t1_val), abs(t2_val), 1)
            if diff / max_val < threshold:
                return 0
            if higher_is_better:
                return 1 if t1_val > t2_val else 2
            else:
                return 1 if t1_val < t2_val else 2

        scorecard = [
            ("Strength", factor_winner(result['t1_elo'], result['t2_elo'])),
            ("Form", factor_winner(result['t1_form'], result['t2_form'])),
            ("Head-to-Head", factor_winner(result['h2h_t1_wins'], result['h2h_t2_wins'])),
            ("Venue Record", factor_winner(result['t1_venue_wr'], result['t2_venue_wr'])),
            ("Home Ground", factor_winner(result['t1_home'], result['t2_home'])),
            ("Batting", factor_winner(result['t1_bat_sr'], result['t2_bat_sr'])),
            ("Bowling", factor_winner(result['t1_bowl_econ'], result['t2_bowl_econ'], higher_is_better=False)),
            ("Experience", factor_winner(result['t1_exp'], result['t2_exp'])),
        ]

        t1_score = sum(1 for _, w in scorecard if w == 1)
        t2_score = sum(1 for _, w in scorecard if w == 2)

        factors = [s[0] for s in scorecard]
        winners = [s[1] for s in scorecard]
        n = len(factors)

        fig_sc, ax_sc = plt.subplots(figsize=(4, 4.5))
        fig_sc.patch.set_facecolor('#0d1117')
        ax_sc.set_facecolor('#0d1117')

        # Title row
        ax_sc.text(0.15, n + 0.8, team1_code, fontsize=12, fontweight='bold',
                   color='#2196F3', ha='center', transform=ax_sc.transData)
        ax_sc.text(0.5, n + 0.8, 'Factor Scorecard', fontsize=8,
                   color='#8b949e', ha='center', transform=ax_sc.transData)
        ax_sc.text(0.85, n + 0.8, team2_code, fontsize=12, fontweight='bold',
                   color='#FF5722', ha='center', transform=ax_sc.transData)

        for i, (fname, winner) in enumerate(scorecard):
            y = n - 1 - i

            # Factor label (center)
            ax_sc.text(0.5, y, fname, fontsize=9, color='#c9d1d9',
                       ha='center', va='center')

            # Dots
            if winner == 1:
                c1, c2 = '#2196F3', '#30363d'
            elif winner == 2:
                c1, c2 = '#30363d', '#FF5722'
            else:
                c1, c2 = '#f7c948', '#f7c948'

            ax_sc.scatter(0.15, y, s=120, color=c1, zorder=5)
            ax_sc.scatter(0.85, y, s=120, color=c2, zorder=5)

            # Separator line
            if i < n - 1:
                ax_sc.axhline(y=y - 0.5, color='#21262d', linewidth=0.5)

        # Score summary at bottom
        ax_sc.axhline(y=-1.0, color='#30363d', linewidth=1)
        ax_sc.text(0.15, -1.6, str(t1_score), fontsize=18, fontweight='bold',
                   color='#2196F3', ha='center', va='center')
        ax_sc.text(0.5, -1.6, f"{t1_score}  -  {t2_score}", fontsize=9,
                   color='#8b949e', ha='center', va='center')
        ax_sc.text(0.85, -1.6, str(t2_score), fontsize=18, fontweight='bold',
                   color='#FF5722', ha='center', va='center')

        ax_sc.set_xlim(-0.05, 1.05)
        ax_sc.set_ylim(-2.5, n + 1.3)
        ax_sc.axis('off')

        plt.tight_layout()
        st.pyplot(fig_sc, use_container_width=True)
        plt.close(fig_sc)
        st.caption("*Colored dot = who has the edge on each factor. Yellow = even.*")

    # --- Why This Team Wins ---
    st.markdown("### Why this prediction?")
    st.caption("Here's what the model looked at — plain-English breakdown of the key factors, including the projected impact-player swap.")

    # Helper: build a comparison bar HTML
    def compare_bar(label, t1_val, t2_val, t1_code, t2_code, higher_is_better=True, fmt=".0f"):
        """Render a visual comparison bar with plain-English verdict."""
        if higher_is_better:
            t1_better = t1_val >= t2_val
        else:
            t1_better = t1_val <= t2_val

        edge_team = t1_code if t1_better else t2_code
        edge_color = "#2196F3" if t1_better else "#FF5722"
        fade_color = "#8b949e"

        # Compute bar widths (normalize to percentages)
        total = abs(t1_val) + abs(t2_val)
        t1_pct = (abs(t1_val) / total * 100) if total > 0 else 50
        t2_pct = 100 - t1_pct

        # Plain-English verdict
        diff = abs(t1_val - t2_val)
        if diff < 0.01 * max(abs(t1_val), abs(t2_val), 1):
            verdict = "Even"
            verdict_color = "#8b949e"
        elif diff < 0.1 * max(abs(t1_val), abs(t2_val), 1):
            verdict = f"Slight edge {edge_team}"
            verdict_color = edge_color
        else:
            verdict = f"{edge_team} has the edge"
            verdict_color = edge_color

        return f"""
        <div style="margin: 14px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-size: 0.85rem; color: #e6edf3; font-weight: 600;">{label}</span>
                <span style="font-size: 0.75rem; color: {verdict_color}; font-weight: 600;">{verdict}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 42px; text-align: right; font-size: 0.8rem; font-weight: 700;
                             color: {'#2196F3' if t1_better else fade_color};">{t1_code}</span>
                <div style="flex: 1; display: flex; height: 26px; border-radius: 6px; overflow: hidden; background: #21262d;">
                    <div style="width: {t1_pct}%; background: {'#2196F3' if t1_better else '#3a4150'};
                                display: flex; align-items: center; justify-content: center;
                                font-size: 0.75rem; font-weight: 700; color: white;">
                        {t1_val:{fmt}}</div>
                    <div style="width: {t2_pct}%; background: {'#FF5722' if not t1_better else '#3a4150'};
                                display: flex; align-items: center; justify-content: center;
                                font-size: 0.75rem; font-weight: 700; color: white;">
                        {t2_val:{fmt}}</div>
                </div>
                <span style="width: 42px; font-size: 0.8rem; font-weight: 700;
                             color: {'#FF5722' if not t1_better else fade_color};">{t2_code}</span>
            </div>
        </div>
        """

    # ----- Section 1: Team Strength -----
    st.markdown("""
    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin: 10px 0;">
        <div style="font-size: 1.1rem; font-weight: 700; color: #e6edf3; margin-bottom: 4px;">
            Team Strength & Track Record
        </div>
        <div style="font-size: 0.8rem; color: #8b949e; margin-bottom: 14px;">
            How strong are these teams based on their overall history?
        </div>
    """, unsafe_allow_html=True)

    # Elo — explain it
    elo_diff = result['t1_elo'] - result['t2_elo']
    if abs(elo_diff) < 30:
        elo_verdict = "Both teams are evenly matched in overall strength."
    elif elo_diff > 0:
        elo_verdict = f"{team1_name} is the stronger team historically."
    else:
        elo_verdict = f"{team2_name} is the stronger team historically."

    st.markdown(
        compare_bar("Overall Strength", result['t1_elo'], result['t2_elo'],
                     team1_code, team2_code, higher_is_better=True, fmt=".0f"),
        unsafe_allow_html=True,
    )
    st.caption(f"*{elo_verdict} (Based on Elo rating — a score that goes up when you win and down when you lose.)*")

    # Form
    t1_f = result['t1_form']
    t2_f = result['t2_form']
    form_parts = []
    if t1_f >= 0.6:
        form_parts.append(f"{team1_code} won {t1_f:.0%} of their last 5 — on a hot streak!")
    elif t1_f <= 0.2:
        form_parts.append(f"{team1_code} won only {t1_f:.0%} of their last 5 — struggling lately.")
    if t2_f >= 0.6:
        form_parts.append(f"{team2_code} won {t2_f:.0%} of their last 5 — on a hot streak!")
    elif t2_f <= 0.2:
        form_parts.append(f"{team2_code} won only {t2_f:.0%} of their last 5 — struggling lately.")
    if not form_parts:
        form_parts.append("Both teams have had a mixed run recently.")

    st.markdown(
        compare_bar("Recent Form (last 5 matches)", t1_f * 100, t2_f * 100,
                     team1_code, team2_code, higher_is_better=True, fmt=".0f"),
        unsafe_allow_html=True,
    )
    st.caption(f"*{' '.join(form_parts)} (Win % in their last 5 matches.)*")

    # H2H
    h2h_t1 = result['h2h_t1_wins']
    h2h_t2 = result['h2h_t2_wins']
    h2h_total = h2h_t1 + h2h_t2
    if h2h_total == 0:
        h2h_text = "These teams haven't played each other before!"
    elif h2h_t1 > h2h_t2:
        h2h_text = f"{team1_name} leads {h2h_t1}-{h2h_t2} in {h2h_total} meetings — they have the psychological edge."
    elif h2h_t2 > h2h_t1:
        h2h_text = f"{team2_name} leads {h2h_t2}-{h2h_t1} in {h2h_total} meetings — they have the psychological edge."
    else:
        h2h_text = f"It's {h2h_t1}-{h2h_t2} in {h2h_total} meetings — nothing to separate them!"

    st.markdown(
        compare_bar("Head-to-Head Record", h2h_t1, h2h_t2,
                     team1_code, team2_code, higher_is_better=True, fmt=".0f"),
        unsafe_allow_html=True,
    )
    st.caption(f"*{h2h_text}*")

    st.markdown("</div>", unsafe_allow_html=True)

    # ----- Section 2: Venue & Toss -----
    st.markdown("""
    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin: 10px 0;">
        <div style="font-size: 1.1rem; font-weight: 700; color: #e6edf3; margin-bottom: 4px;">
            Venue & Toss Factors
        </div>
        <div style="font-size: 0.8rem; color: #8b949e; margin-bottom: 14px;">
            Does the ground or toss give either team an advantage?
        </div>
    """, unsafe_allow_html=True)

    # Home advantage
    home_team_name = team1_name if result["t1_home"] else (team2_name if result["t2_home"] else None)
    if home_team_name:
        home_text = f"{home_team_name} is playing at HOME — crowd support, familiar conditions, and comfort factor."
        home_icon = "🏠"
    else:
        home_text = "Neutral venue — no home advantage for either team."
        home_icon = "🏟"

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin: 10px 0; padding: 12px 16px;
                background: #0d1117; border-radius: 8px; border: 1px solid #30363d;">
        <span style="font-size: 1.8rem;">{home_icon}</span>
        <div>
            <div style="font-size: 0.9rem; font-weight: 600; color: #e6edf3;">Home Ground</div>
            <div style="font-size: 0.8rem; color: #b1bac4;">{home_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Venue win rates
    st.markdown(
        compare_bar("Win Rate at This Venue", result['t1_venue_wr'] * 100, result['t2_venue_wr'] * 100,
                     team1_code, team2_code, higher_is_better=True, fmt=".0f"),
        unsafe_allow_html=True,
    )
    st.caption(f"*How often each team wins when playing at {venue}.*")

    # Bat first advantage
    bfp = result['bat_first_pct']
    toss_chose = toss_decision
    if bfp > 0.55:
        bat_text = f"Batting first wins {bfp:.0%} of matches here — this ground favours setting a target."
    elif bfp < 0.45:
        bat_text = f"Chasing teams win {1-bfp:.0%} of matches here — this ground favours bowling first."
    else:
        bat_text = f"It's roughly 50-50 between batting first and chasing here ({bfp:.0%} bat-first wins)."

    toss_who = team1_name if toss_winner == team1_name else team2_name
    toss_text = f"{toss_who} won the toss and chose to **{toss_chose}**."

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin: 10px 0; padding: 12px 16px;
                background: #0d1117; border-radius: 8px; border: 1px solid #30363d;">
        <span style="font-size: 1.8rem;">🪙</span>
        <div>
            <div style="font-size: 0.9rem; font-weight: 600; color: #e6edf3;">Toss & Pitch</div>
            <div style="font-size: 0.8rem; color: #b1bac4;">{toss_text} {bat_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ----- Section 3: Squad Strength -----
    st.markdown("""
    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin: 10px 0;">
        <div style="font-size: 1.1rem; font-weight: 700; color: #e6edf3; margin-bottom: 4px;">
            Squad Power Comparison
        </div>
        <div style="font-size: 0.8rem; color: #8b949e; margin-bottom: 14px;">
            How do the selected squads compare? Based on the XI, adjusted for the projected impact substitute in the innings where that rule is most likely to matter.
        </div>
    """, unsafe_allow_html=True)

    # Batting
    bat_edge = team1_code if result['t1_bat_sr'] >= result['t2_bat_sr'] else team2_code
    bat_diff = abs(result['t1_bat_sr'] - result['t2_bat_sr'])
    if bat_diff < 3:
        bat_text = "Both batting lineups are equally strong — expect a balanced contest."
    elif bat_edge == team1_code:
        bat_text = f"{team1_name}'s batters score faster on average — they can accelerate better in the death overs."
    else:
        bat_text = f"{team2_name}'s batters score faster on average — they can accelerate better in the death overs."

    st.markdown(
        compare_bar("Batting Power (Strike Rate)", result['t1_bat_sr'], result['t2_bat_sr'],
                     team1_code, team2_code, higher_is_better=True, fmt=".1f"),
        unsafe_allow_html=True,
    )
    st.caption(f"*{bat_text} (Higher strike rate = more runs per ball.)*")

    # Bowling — lower is better
    bowl_edge = team1_code if result['t1_bowl_econ'] <= result['t2_bowl_econ'] else team2_code
    bowl_diff = abs(result['t1_bowl_econ'] - result['t2_bowl_econ'])
    if bowl_diff < 0.3:
        bowl_text = "Both bowling attacks are equally tight — no clear advantage."
    elif bowl_edge == team1_code:
        bowl_text = f"{team1_name}'s bowlers give away fewer runs per over — tighter bowling attack."
    else:
        bowl_text = f"{team2_name}'s bowlers give away fewer runs per over — tighter bowling attack."

    st.markdown(
        compare_bar("Bowling Tightness (Economy)", result['t1_bowl_econ'], result['t2_bowl_econ'],
                     team1_code, team2_code, higher_is_better=False, fmt=".1f"),
        unsafe_allow_html=True,
    )
    st.caption(f"*{bowl_text} (Lower economy = fewer runs conceded per over.)*")

    # Experience
    exp_edge = team1_code if result['t1_exp'] >= result['t2_exp'] else team2_code
    exp_diff = abs(result['t1_exp'] - result['t2_exp'])
    if exp_diff < 10:
        exp_text = "Both squads have similar experience levels."
    elif exp_edge == team1_code:
        exp_text = f"{team1_name} has a much more experienced squad — that matters in pressure situations."
    else:
        exp_text = f"{team2_name} has a much more experienced squad — that matters in pressure situations."

    st.markdown(
        compare_bar("Experience (Avg Matches Played)", result['t1_exp'], result['t2_exp'],
                     team1_code, team2_code, higher_is_better=True, fmt=".0f"),
        unsafe_allow_html=True,
    )
    st.caption(f"*{exp_text} (Average IPL matches played per player in the XI.)*")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="section-kicker">Saved Data</div>', unsafe_allow_html=True)
st.markdown('<div class="page-section-title">Account Workspace</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-section-subtitle">Review saved predictions, manage your reusable Playing XI combinations, and browse all current team squads.</div>',
    unsafe_allow_html=True,
)

dashboard_tab, squads_tab = st.tabs(["My Dashboard", "All Squads"])
with dashboard_tab:
    render_user_dashboard(current_user)
with squads_tab:
    render_all_squads_browser()

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption("Built with XGBoost | Data: IPL 2008-2025 (1,169 matches) | Player names must match cricsheet format")
