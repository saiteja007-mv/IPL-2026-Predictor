# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IPL 2026 match winner prediction system — fresh start. The goal is to build an ML pipeline trained on 18 seasons of historical IPL data (2008–2025) to predict match outcomes for the 2026 season.

## Current State

This is a greenfield rebuild. The only files present are:

- `training.ipynb` — empty notebook (starting point for the ML pipeline)
- `Datasets/` — raw data files ready to use
- `IPL Team logos/` — team logo PNGs (10 teams + processed versions)

No models, no app, no scripts exist yet — everything needs to be built.

## Datasets

All source data is in `Datasets/`:

| File | Description |
|------|-------------|
| `matches.csv` | IPL match results 2008–2025 (season, teams, venue, toss, winner, result) |
| `ball_by_ball_data.csv` | Aggregated ball-by-ball summary stats (2008–2025, ~278K deliveries, 1169 matches) |
| `player_ipl_stats.csv` | Per-player IPL career aggregates: batting SR, bowling econ, experience, boundaries |
| `player_lifetime_stats.csv` | Cross-league T20 career stats (international, IPL, BBL, PSL, Hundred, etc.) — use for uncapped/new players |
| `players-data-updated.csv` | Player metadata: bat/bowl style, field position, full name, player_id |
| `team_aliases.csv` | Maps all historical team name variants to canonical short codes (RCB, MI, CSK, etc.) |
| `cricsheet_cache/all_male_json.zip` | Cached cricsheet.org download for rebuilding player stats |

### Key Schema Notes

- `matches.csv` columns: `match_id, season_id, city, date, venue, toss_winner, team1, team2, toss_decision, winner, win_by_runs, win_by_wickets, player_of_match, result`
- `player_ipl_stats.csv` columns: `player, matches_batted, total_runs, balls_faced, fours, sixes, batting_sr, avg_runs, boundary_pct, matches_bowled, balls_bowled, runs_conceded, wickets, overs, bowling_econ, bowling_sr, ipl_experience`
- `player_lifetime_stats.csv` columns: `player_id, player_name, bat_style, bowl_style` + per-league stats (overall, international, national, domestic, ipl, psl, hundred, bbl, t10, other)
- `team_aliases.csv`: always resolve team names through this before any lookup — covers defunct teams (RPS, GL, KTK, PW) and spelling variants

## IPL 2026 Teams

| Code | Team | Home Ground |
|------|------|-------------|
| MI | Mumbai Indians | Wankhede Stadium |
| CSK | Chennai Super Kings | MA Chidambaram Stadium |
| RCB | Royal Challengers Bengaluru | M Chinnaswamy Stadium |
| KKR | Kolkata Knight Riders | Eden Gardens |
| DC | Delhi Capitals | Arun Jaitley Stadium |
| SRH | Sunrisers Hyderabad | Rajiv Gandhi International Stadium |
| RR | Rajasthan Royals | Sawai Mansingh Stadium |
| PK | Punjab Kings | Punjab Cricket Association Stadium |
| LSG | Lucknow Super Giants | BRSABV Ekana Cricket Stadium |
| GT | Gujarat Titans | Narendra Modi Stadium |

## Design Principles for New Code

- Use `team_aliases.csv` for all team name normalisation — never hardcode variants
- Prefer `player_lifetime_stats.csv` as fallback for players absent from `player_ipl_stats.csv`
- Keep a centralised `project_paths.py` so all scripts/notebooks import paths from one place
- Save trained model artifacts to a `Models/` directory
- Append new 2026 match results to `matches.csv` after each game; re-run feature engineering to update Elo/form
