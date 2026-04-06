"""
IPL 2026 Daily Match Updater
=============================
Fetches completed IPL 2026 match results from CricAPI,
appends new matches to matches.csv, stores scores for NRR,
retrains the XGBoost model, and updates all artifacts.

Usage:
    python update_matches.py              # Update + retrain
    python update_matches.py --backfill   # Fetch ALL completed IPL 2026 matches
    python update_matches.py --dry-run    # Preview without writing files
    python update_matches.py --no-retrain # Update data only, skip model retrain

API: api.cricapi.com (free tier: 100 requests/day)
"""

import os
import re
import sys
import json
import pickle
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "Datasets"
MODELS_DIR = BASE_DIR / "Models"
MATCHES_CSV = DATASETS_DIR / "matches.csv"
SCORES_JSON = DATASETS_DIR / "ipl_2026_scores.json"
ELO_PKL = MODELS_DIR / "elo_ratings.pkl"
HISTORY_PKL = MODELS_DIR / "match_history.pkl"
UPDATE_LOG = BASE_DIR / "Datasets" / "update_log.json"

API_BASE = "https://api.cricapi.com/v1"
API_KEY = os.environ.get("CRICAPI_KEY", "7cd7438d-0e3b-4a25-8520-b34baff42e77")
IPL_2026_SERIES_ID = "87c62aac-bc3c-4738-ab93-19da0690488f"

# Elo K-factor (must match training.ipynb)
K_FACTOR = 32

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── API Helpers ────────────────────────────────────────────────
def api_get(endpoint: str, params: dict | None = None) -> dict:
    """Make a GET request to CricAPI with automatic key injection."""
    params = params or {}
    params["apikey"] = API_KEY
    url = f"{API_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"API error: {data}")
    hits = data.get("info", {})
    log.info(f"  API {endpoint}: hits {hits.get('hitsToday', '?')}/{hits.get('hitsLimit', '?')}")
    return data


def fetch_completed_matches() -> list[dict]:
    """Fetch all completed IPL 2026 matches from the series endpoint."""
    data = api_get("series_info", {"id": IPL_2026_SERIES_ID})
    match_list = data["data"].get("matchList", [])

    # Deduplicate by match ID (API returns dupes across pages)
    seen = set()
    completed = []
    for m in match_list:
        if m.get("matchEnded") and m["id"] not in seen:
            seen.add(m["id"])
            completed.append(m)

    completed.sort(key=lambda x: x["date"])
    log.info(f"Found {len(completed)} completed IPL 2026 matches")
    return completed


def fetch_match_detail(match_id: str) -> dict:
    """Fetch detailed match info (toss, scores, winner) for a single match."""
    data = api_get("match_info", {"id": match_id})
    return data["data"]


# ── Alias Resolution ──────────────────────────────────────────
def load_alias_map() -> dict:
    """Load alias_map from pkl."""
    with open(MODELS_DIR / "alias_map.pkl", "rb") as f:
        return pickle.load(f)


def resolve_team(name: str, alias_map: dict) -> str:
    """Resolve a team name to its canonical code."""
    return alias_map.get(name.strip().lower(), name.strip())


# ── Win Margin Parsing ────────────────────────────────────────
def parse_win_margin(status: str) -> tuple[float, float]:
    """
    Parse 'Team won by X runs' or 'Team won by X wkts' from status string.
    Returns (win_by_runs, win_by_wickets).
    """
    runs_match = re.search(r"won by (\d+) run", status, re.IGNORECASE)
    wkts_match = re.search(r"won by (\d+) wkt", status, re.IGNORECASE)
    if runs_match:
        return float(runs_match.group(1)), 0.0
    elif wkts_match:
        return 0.0, float(wkts_match.group(1))
    return 0.0, 0.0


# ── CSV Row Builder ───────────────────────────────────────────
def build_csv_row(detail: dict, match_number: int) -> dict:
    """Convert API match detail to a matches.csv row."""
    teams = detail["teams"]
    team1, team2 = teams[0], teams[1]

    # Determine city from venue
    venue = detail.get("venue", "")
    city = venue.split(",")[-1].strip() if "," in venue else venue.split(" ")[0]

    # Toss info (API returns lowercase)
    toss_winner_raw = detail.get("tossWinner", "")
    toss_decision = detail.get("tossChoice", "")
    # Normalize toss_decision: API returns "bat"/"bowl", CSV uses "bat"/"field"
    if toss_decision == "bowl":
        toss_decision = "field"

    # Match winner
    match_winner = detail.get("matchWinner", "")

    # Win margin
    status = detail.get("status", "")
    win_by_runs, win_by_wickets = parse_win_margin(status)

    # Player of match (may not be in match_info, use empty if absent)
    pom = detail.get("matchPlayerOfMatch", {}).get("name", "")
    if not pom:
        pom = detail.get("playerOfMatch", "")

    # Find toss winner full name (API returns lowercase)
    toss_winner = ""
    for t in [team1, team2]:
        if t.lower() == toss_winner_raw.lower():
            toss_winner = t
            break

    # Generate a match_id for 2026 (start from 2026001)
    match_id = 2026000 + match_number

    return {
        "match_id": match_id,
        "season_id": 2026,
        "balls_per_over": 6,
        "city": city,
        "date": detail.get("date", ""),
        "event_name": "Indian Premier League",
        "match_number": float(match_number),
        "gender": "male",
        "match_type": "T20",
        "format": "T20",
        "overs": 20,
        "season": 2026,
        "team_type": "club",
        "venue": venue,
        "toss_winner": toss_winner,
        "team1": team1,
        "team2": team2,
        "toss_decision": toss_decision,
        "winner": match_winner,
        "win_by_runs": win_by_runs if win_by_runs else "",
        "win_by_wickets": win_by_wickets if win_by_wickets else "",
        "player_of_match": pom,
        "result": "win",
    }


# ── Elo Update ────────────────────────────────────────────────
def update_elo(winner_elo: float, loser_elo: float, k: int = K_FACTOR) -> tuple[float, float]:
    """Calculate new Elo ratings after a match (mirrors training.ipynb)."""
    expected_win = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    new_winner = winner_elo + k * (1 - expected_win)
    new_loser = loser_elo + k * (0 - (1 - expected_win))
    return new_winner, new_loser


# ── Match Number Extraction ───────────────────────────────────
def extract_match_number(name: str) -> int:
    """Extract match number from API match name like '8th Match'."""
    m = re.search(r"(\d+)(?:st|nd|rd|th) Match", name)
    return int(m.group(1)) if m else 0


# ── Main Pipeline ─────────────────────────────────────────────
def run_update(backfill: bool = False, dry_run: bool = False, no_retrain: bool = False):
    """
    Main update pipeline:
    1. Load existing data
    2. Fetch completed IPL 2026 matches
    3. Identify new matches not yet in matches.csv
    4. Fetch detailed info for each new match (toss, winner, scores)
    5. Append to matches.csv
    6. Update elo_ratings.pkl and match_history.pkl incrementally
    """
    # Load existing state
    df_existing = pd.read_csv(MATCHES_CSV)
    with open(ELO_PKL, "rb") as f:
        elo_ratings = pickle.load(f)
    with open(HISTORY_PKL, "rb") as f:
        match_history = pickle.load(f)
    alias_map = load_alias_map()

    log.info(f"Existing: {len(df_existing)} matches in CSV, {len(match_history)} in history")
    log.info(f"Current Elo top-5: {sorted(elo_ratings.items(), key=lambda x: -x[1])[:5]}")

    # Load update log to track which API match IDs we've already processed
    processed_ids = set()
    if UPDATE_LOG.exists():
        with open(UPDATE_LOG) as f:
            log_data = json.load(f)
            processed_ids = set(log_data.get("processed_match_ids", []))

    # Fetch completed matches from API
    completed = fetch_completed_matches()

    # Filter to only new matches
    if not backfill:
        new_matches = [m for m in completed if m["id"] not in processed_ids]
    else:
        new_matches = [m for m in completed if m["id"] not in processed_ids]
        if not new_matches:
            log.info("Backfill mode: all matches already processed")

    if not new_matches:
        log.info("No new matches to process. Everything is up to date!")
        return 0

    log.info(f"Processing {len(new_matches)} new matches...")

    # Fetch detailed info for each new match
    new_rows = []
    new_scores = []
    new_history_entries = []
    api_hits = 1  # Already used 1 for series_info

    for match in new_matches:
        match_num = extract_match_number(match["name"])
        log.info(f"  Fetching Match #{match_num}: {match['name'][:60]}...")

        detail = fetch_match_detail(match["id"])
        api_hits += 1

        # Check for no-result / abandoned
        if not detail.get("matchWinner"):
            log.warning(f"  Skipping {match['name'][:40]} — no winner (abandoned/no result)")
            processed_ids.add(match["id"])
            continue

        row = build_csv_row(detail, match_num)
        new_rows.append(row)

        # Store score data for NRR calculation
        scores = detail.get("score", [])
        score_entry = {
            "match_num": match_num,
            "date": detail.get("date", ""),
            "team1": detail["teams"][0],
            "team2": detail["teams"][1],
            "winner": detail.get("matchWinner", ""),
            "innings": [],
        }
        for inning in scores:
            score_entry["innings"].append({
                "team": inning.get("inning", "").replace(" Inning 1", "").replace(" Inning 2", ""),
                "runs": inning.get("r", 0),
                "wickets": inning.get("w", 0),
                "overs": inning.get("o", 0),
            })
        new_scores.append(score_entry)

        # Build history entry (canonical codes)
        winner_code = resolve_team(row["winner"], alias_map)
        team1_code = resolve_team(row["team1"], alias_map)
        team2_code = resolve_team(row["team2"], alias_map)
        toss_code = resolve_team(row["toss_winner"], alias_map)

        history_entry = {
            "team1": team1_code,
            "team2": team2_code,
            "winner": winner_code,
            "venue": row["venue"],
            "toss_winner": toss_code,
            "toss_decision": row["toss_decision"],
        }
        new_history_entries.append((history_entry, winner_code, team1_code, team2_code))

        processed_ids.add(match["id"])

    if not new_rows:
        log.info("No valid new matches with results to add.")
        return 0

    if dry_run:
        log.info(f"\n[DRY RUN] Would add {len(new_rows)} matches:")
        for r in new_rows:
            log.info(f"  {r['date']} | {r['team1']} vs {r['team2']} | Winner: {r['winner']}")
        log.info(f"  API hits used: {api_hits}")
        return len(new_rows)

    # Append to matches.csv
    df_new = pd.DataFrame(new_rows)
    df_new.to_csv(MATCHES_CSV, mode="a", header=False, index=False)
    log.info(f"Appended {len(new_rows)} rows to {MATCHES_CSV}")

    # Update Elo ratings and match history incrementally
    for entry, winner_code, team1_code, team2_code in new_history_entries:
        loser_code = team2_code if winner_code == team1_code else team1_code

        # Ensure both teams have Elo entries
        elo_ratings.setdefault(winner_code, 1500)
        elo_ratings.setdefault(loser_code, 1500)

        old_w = elo_ratings[winner_code]
        old_l = elo_ratings[loser_code]
        new_w, new_l = update_elo(old_w, old_l)
        elo_ratings[winner_code] = new_w
        elo_ratings[loser_code] = new_l

        match_history.append(entry)
        log.info(
            f"  Elo: {winner_code} {old_w:.0f}→{new_w:.0f} | "
            f"{loser_code} {old_l:.0f}→{new_l:.0f}"
        )

    # Save updated pickles
    with open(ELO_PKL, "wb") as f:
        pickle.dump(elo_ratings, f)
    with open(HISTORY_PKL, "wb") as f:
        pickle.dump(match_history, f)
    log.info(f"Saved updated elo_ratings.pkl ({len(elo_ratings)} teams)")
    log.info(f"Saved updated match_history.pkl ({len(match_history)} matches)")

    # Save scores for NRR calculation
    existing_scores = []
    if SCORES_JSON.exists():
        try:
            with open(SCORES_JSON) as f:
                existing_scores = json.load(f)
        except Exception:
            pass
    existing_scores.extend(new_scores)
    with open(SCORES_JSON, "w") as f:
        json.dump(existing_scores, f, indent=2)
    log.info(f"Saved {len(existing_scores)} match scores to {SCORES_JSON}")

    # Save update log
    log_data = {
        "processed_match_ids": list(processed_ids),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_2026_matches": len(new_rows) + len(processed_ids) - len(new_rows),
        "api_hits_used": api_hits,
    }
    with open(UPDATE_LOG, "w") as f:
        json.dump(log_data, f, indent=2)
    log.info(f"Update log saved to {UPDATE_LOG}")

    # Print summary
    log.info("\n" + "=" * 60)
    log.info("UPDATE COMPLETE")
    log.info(f"  New matches added: {len(new_rows)}")
    log.info(f"  Total matches in CSV: {len(df_existing) + len(new_rows)}")
    log.info(f"  Total match history: {len(match_history)}")
    log.info(f"  API hits used today: {api_hits}")
    log.info("=" * 60)

    # Retrain model if requested
    if not no_retrain:
        log.info("Retraining model with updated data...")
        try:
            from retrain_model import retrain, save_artifacts
            result = retrain(quiet=False)
            if result:
                save_artifacts(*result, quiet=False)
                log.info("Model retrained successfully!")
            else:
                log.error("Model retrain failed!")
        except Exception as e:
            log.error(f"Retrain error: {e}")

    return len(new_rows)


if __name__ == "__main__":
    backfill = "--backfill" in sys.argv
    dry_run = "--dry-run" in sys.argv
    no_retrain = "--no-retrain" in sys.argv
    run_update(backfill=backfill, dry_run=dry_run, no_retrain=no_retrain)
