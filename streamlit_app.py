"""
IPL 2026 Match Predictor — Streamlit Web App

Run: streamlit run app.py
Requires: trained model artifacts in Models/ (run training.ipynb first)
"""

import os
import json
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
from difflib import SequenceMatcher

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
</style>
""", unsafe_allow_html=True)

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
    "SRH": {"Pat Cummins", "Travis Head", "Heinrich Klaasen", "Kamindu Mendis", "Brydon Carse", "Liam Livingstone", "Jack Edwards"},
    "RR": {"Sam Curran", "Donovan Ferreira", "Lhuan-Dre Pretorius", "Shimron Hetmyer", "Jofra Archer", "Kwena Maphaka", "Nandre Burger", "Adam Milne"},
    "PK": {"Marcus Stoinis", "Azmatullah Omarzai", "Marco Jansen", "Mitchell Owen", "Xavier Bartlett", "Lockie Ferguson", "Cooper Connolly", "Ben Dwarshuis"},
    "LSG": {"Aiden Markram", "Matthew Breetzke", "Nicholas Pooran", "Mitchell Marsh", "Wanindu Hasaranga", "Anrich Nortje", "Josh Inglis"},
    "GT": {"Jos Buttler", "Glenn Phillips", "Kagiso Rabada", "Rashid Khan", "Jason Holder", "Tom Banton", "Luke Wood"},
}

LOGO_DIR = "IPL Team logos/processed"


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
        }

    lifetime = {}
    for _, row in player_lifetime.iterrows():
        lifetime[row["player_name"]] = {
            "batting_sr": row["overall_batting_sr"] if pd.notna(row["overall_batting_sr"]) else 0,
            "bowling_econ": row["overall_bowling_econ"] if pd.notna(row["overall_bowling_econ"]) else 0,
            "experience": row["overall_matches"] if pd.notna(row["overall_matches"]) else 0,
            "avg_runs": row["overall_batting_avg"] if pd.notna(row["overall_batting_avg"]) else 0,
        }

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
DEFAULT_STATS = {"batting_sr": 120.0, "bowling_econ": 8.0, "experience": 0, "avg_runs": 20.0}


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


def predict_match(
    team1_code, team2_code, venue, toss_winner_code, toss_decision,
    team1_xi, team2_xi, model, feature_cols, elo_ratings, match_history,
    ipl_stats, lifetime_stats,
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

    # Player strength
    t1_stats = [get_player_stats(p, ipl_stats, lifetime_stats) for p in team1_xi]
    t2_stats = [get_player_stats(p, ipl_stats, lifetime_stats) for p in team2_xi]

    t1_bat_sr = np.mean([s["batting_sr"] for s in t1_stats])
    t1_bowl_econ = np.mean([s["bowling_econ"] for s in t1_stats])
    t1_exp = np.mean([s["experience"] for s in t1_stats])
    t1_avg_runs = np.mean([s["avg_runs"] for s in t1_stats])

    t2_bat_sr = np.mean([s["batting_sr"] for s in t2_stats])
    t2_bowl_econ = np.mean([s["bowling_econ"] for s in t2_stats])
    t2_exp = np.mean([s["experience"] for s in t2_stats])
    t2_avg_runs = np.mean([s["avg_runs"] for s in t2_stats])

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
    }


# ============================================================
# Load everything
# ============================================================
if not os.path.exists("Models/xgb_model.pkl"):
    st.error("Model not found! Run `training.ipynb` first to train and save the model.")
    st.stop()

model, feature_cols, elo_ratings, alias_map, match_history = load_model()
ipl_stats, lifetime_stats = load_player_stats()
full_to_cricsheet, players_meta = build_name_resolver()

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
st.markdown("---")

# ============================================================
# UI — Match Setup (Sidebar)
# ============================================================
with st.sidebar:
    st.header("Match Setup")

    team_names = list(TEAMS.keys())

    team1_name = st.selectbox("Team 1", team_names, index=0)
    team2_name = st.selectbox("Team 2", team_names, index=1)

    if team1_name == team2_name:
        st.warning("Please select two different teams.")
        st.stop()

    venue = st.selectbox("Venue", VENUES)
    toss_winner = st.selectbox("Toss Winner", [team1_name, team2_name])
    toss_decision = st.selectbox("Toss Decision", ["field", "bat"])

    st.markdown("---")
    st.caption("Enter Playing XI below (one name per line)")

# ============================================================
# UI — Playing XI Input
# ============================================================
team1_code = TEAMS[team1_name]
team2_code = TEAMS[team2_name]

col_xi1, col_xi2 = st.columns(2)

# Get squad lists for selected teams
squad1 = SQUADS_2026.get(team1_code, [])
squad2 = SQUADS_2026.get(team2_code, [])
overseas1 = OVERSEAS_PLAYERS.get(team1_code, set())
overseas2 = OVERSEAS_PLAYERS.get(team2_code, set())


def render_xi_selector(col, team_name, team_code, squad, overseas_set, key_suffix):
    """Render the Playing XI selector with overseas tracking."""
    with col:
        logo_path = f"{LOGO_DIR}/{team_code}.png"
        c_logo, c_name = st.columns([0.3, 2])
        if os.path.exists(logo_path):
            c_logo.image(logo_path, width=55)
        c_name.subheader(team_name)

        if squad:
            # Format options: add flag for overseas players
            display_options = [f"{p}  🌍" if p in overseas_set else p for p in squad]
            option_map = dict(zip(display_options, squad))  # display -> real name

            selected_display = st.multiselect(
                f"Select Playing XI ({team_code})",
                options=display_options,
                default=[],
                max_selections=11,
                key=f"xi_{key_suffix}_select",
            )
            selected_real = [option_map[d] for d in selected_display]

            # Count overseas in selection
            overseas_count = sum(1 for p in selected_real if p in overseas_set)
            indian_count = len(selected_real) - overseas_count

            # Status line
            status_parts = []
            status_parts.append(f"{len(selected_real)}/11 players")
            if overseas_count > 4:
                status_parts.append(f"⚠️ {overseas_count}/4 overseas (max 4!)")
            else:
                status_parts.append(f"🌍 {overseas_count}/4 overseas")
            status_parts.append(f"🇮🇳 {indian_count} Indian")

            if overseas_count > 4:
                st.error(f"Too many overseas players! You selected {overseas_count} — maximum is 4.")
            else:
                st.caption(" · ".join(status_parts))

            return selected_real, overseas_count
        else:
            xi_text = st.text_area(
                f"Playing XI — {team_code}",
                height=280,
                placeholder="Player 1\nPlayer 2\n...",
                label_visibility="collapsed",
            )
            xi = [n.strip() for n in xi_text.strip().split("\n") if n.strip()]
            return xi, 0


team1_xi, t1_overseas_count = render_xi_selector(col_xi1, team1_name, team1_code, squad1, overseas1, "1")
team2_xi, t2_overseas_count = render_xi_selector(col_xi2, team2_name, team2_code, squad2, overseas2, "2")

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
    if len(team2_xi) == 0:
        errors.append(f"Enter Playing XI for {team2_name}")
    elif len(team2_xi) != 11:
        errors.append(f"{team2_name}: selected {len(team2_xi)} players (need exactly 11)")
    if t2_overseas_count > 4:
        errors.append(f"{team2_name}: {t2_overseas_count} overseas players selected (max 4 allowed)")

    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    # --- Resolve player names ---
    # Convert full names (e.g. "Virat Kohli") to cricsheet format ("V Kohli")
    resolved_t1 = []
    resolved_t2 = []
    resolved_names_t1 = []  # for display
    resolved_names_t2 = []
    unknown_t1 = []
    unknown_t2 = []

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

    # Show name resolutions
    if resolved_names_t1:
        st.info(f"**{team1_code}** name matches: " + ", ".join(resolved_names_t1))
    if resolved_names_t2:
        st.info(f"**{team2_code}** name matches: " + ", ".join(resolved_names_t2))

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
    )

    t1_prob = result["t1_prob"]
    t2_prob = result["t2_prob"]
    winner_name = team1_name if t1_prob > 0.5 else team2_name
    winner_code = team1_code if t1_prob > 0.5 else team2_code
    winner_prob = max(t1_prob, t2_prob)

    st.markdown("---")

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
    st.caption("Here's what the model looked at — plain-English breakdown of the key factors.")

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
            How do the selected playing XIs compare? Based on career stats of the 11 players.
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

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption("Built with XGBoost | Data: IPL 2008-2025 (1,169 matches) | Player names must match cricsheet format")
