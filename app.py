import os
import math
from typing import Dict, Any

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    scoreboardv2,
    playergamelog,
    leaguedashteamstats,
    leaguedashplayerstats,
    teamgamelog,
    teamdashboardbygeneralsplits,
)

DEFAULT_SEASON = os.getenv("DEFAULT_SEASON", "2025-26")

app = FastAPI(title="NBA Stats Wrapper", version="1.2.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ----------------------------
# NBA headers (cloud-friendly)
# ----------------------------
NBA_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://stats.nba.com/",
    "Origin": "https://stats.nba.com",
    "Connection": "keep-alive",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

# ----------------------------
# Normal CDF + PPF (NO SCIPY)
# ----------------------------
def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def normal_ppf(p: float) -> float:
    # Acklam approximation (good enough for p10/p90)
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

def normal_ppf_loc_scale(p: float, mean: float, sd: float) -> float:
    return mean + sd * normal_ppf(p)

# ----------------------------
# Odds / no-vig helpers
# ----------------------------
def american_to_implied_prob(odds: int) -> float:
    if odds == 0:
        raise ValueError("odds cannot be 0")
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)

def no_vig_two_way(p_over: float, p_under: float) -> Dict[str, float]:
    s = p_over + p_under
    if s <= 0:
        return {"p_over": 0.5, "p_under": 0.5, "vig": 0.0}
    return {"p_over": p_over / s, "p_under": p_under / s, "vig": max(0.0, s - 1.0)}

# ----------------------------
# nba_api -> DataFrame helper
# ----------------------------
def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def gamelog_df(player_id: int, season: str) -> pd.DataFrame:
    """
    Pull player logs and convert to a dataframe.
    Any stats.nba.com block/timeout becomes a readable 502.
    """
    try:
        gl = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            timeout=60,
            headers=NBA_HEADERS,
        )
        d = gl.get_dict()
        rs = d.get("resultSets", [])
        if not rs:
            return pd.DataFrame()
        headers = rs[0]["headers"]
        rows = rs[0]["rowSet"]
        return pd.DataFrame(rows, columns=headers)
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"
        raise HTTPException(status_code=502, detail=f"NBA stats fetch failed: {msg[:300]}")

# ----------------------------
# Pro baseline model: Minutes × Rate
# ----------------------------
STAT_MAP = {"PTS": "PTS", "REB": "REB", "AST": "AST", "FG3M": "FG3M"}

def estimate_minutes(df: pd.DataFrame, last_n: int = 10) -> Dict[str, float]:
    if df.empty:
        return {"mean": 0.0, "sd": 8.0}

    mins = df["MIN"].tail(last_n).apply(safe_float).to_numpy()
    if len(mins) == 0:
        return {"mean": 0.0, "sd": 8.0}

    mean = float(np.mean(mins))
    sd = float(np.std(mins, ddof=1)) if len(mins) > 1 else 6.0
    sd = float(np.clip(sd, 3.0, 12.0))
    return {"mean": mean, "sd": sd}

def estimate_rate_per_minute(df: pd.DataFrame, stat_col: str, last_n: int = 15) -> Dict[str, float]:
    if df.empty:
        return {"mean": 0.0, "sd": 0.05}

    tail = df.tail(last_n).copy()
    tail["MIN_f"] = tail["MIN"].apply(safe_float)
    tail[stat_col] = tail[stat_col].apply(safe_float)

    tail = tail[tail["MIN_f"] > 0]
    if tail.empty:
        return {"mean": 0.0, "sd": 0.05}

    rates = (tail[stat_col] / tail["MIN_f"]).to_numpy()
    mean = float(np.mean(rates))
    sd = float(np.std(rates, ddof=1)) if len(rates) > 1 else max(0.03, 0.15 * mean)
    sd = float(np.clip(sd, 0.02, 0.25))
    return {"mean": mean, "sd": sd}

def combine_minutes_rate(mins: Dict[str, float], rate: Dict[str, float]) -> Dict[str, float]:
    # X = M * R, approximate variance assuming independence
    m_mu, m_sd = mins["mean"], mins["sd"]
    r_mu, r_sd = rate["mean"], rate["sd"]

    mean = m_mu * r_mu
    var = (m_mu**2) * (r_sd**2) + (r_mu**2) * (m_sd**2) + (m_sd**2) * (r_sd**2)
    sd = math.sqrt(max(1e-9, var))
    sd = float(np.clip(sd, 1.0, 15.0))
    return {"mean": float(mean), "sd": float(sd)}

def prob_over(line: float, mean: float, sd: float) -> float:
    if sd <= 0:
        return 1.0 if mean > line else 0.0
    z = (mean - line) / sd
    return float(normal_cdf(z))

def confidence_label(minutes_sd: float, stat_sd: float, n_games: int) -> str:
    risk = 0
    if n_games < 8:
        risk += 2
    if minutes_sd >= 9:
        risk += 2
    elif minutes_sd >= 7:
        risk += 1
    if stat_sd >= 10:
        risk += 2
    elif stat_sd >= 7:
        risk += 1
    if risk <= 1:
        return "high"
    if risk <= 3:
        return "medium"
    return "low"

def unit_sizing(edge_pct: float, confidence: str) -> Dict[str, Any]:
    # Conservative, pro-ish unit buckets
    if edge_pct < 2.0 or confidence == "low":
        return {"recommendation": "NO_BET", "units": 0.0}
    if edge_pct < 3.5:
        return {"recommendation": "PLAY", "units": 0.5}
    if edge_pct < 6.0:
        return {"recommendation": "PLAY", "units": 1.0}
    return {"recommendation": "PLAY", "units": 1.5}

# ----------------------------
# Endpoints: health + lookup
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True, "default_season": DEFAULT_SEASON}

@app.get("/players/search")
def search_players(q: str):
    q = (q or "").strip()
    if len(q) < 2:
        raise HTTPException(status_code=400, detail="q must be at least 2 characters")
    return {"query": q, "results": players.find_players_by_full_name(q)[:10]}

@app.get("/meta/player/by_name")
def player_by_name(name: str):
    name = (name or "").strip()
    if len(name) < 2:
        raise HTTPException(status_code=400, detail="name must be at least 2 characters")
    results = players.find_players_by_full_name(name)
    if not results:
        raise HTTPException(status_code=404, detail=f"No players found for '{name}'")
    return {"best_match": results[0], "top_matches": results[:10]}

@app.get("/teams")
def list_teams():
    return {"results": teams.get_teams()}

@app.get("/meta/team/by_abbr")
def team_by_abbr(abbr: str):
    abbr = (abbr or "").strip().upper()
    for t in teams.get_teams():
        if t.get("abbreviation") == abbr:
            return t
    raise HTTPException(status_code=404, detail=f"Team not found for abbreviation '{abbr}'")

# ----------------------------
# Endpoints: NBA stats calls
# ----------------------------
@app.get("/scoreboard")
def get_scoreboard(game_date: str):
    game_date = (game_date or "").strip()
    if len(game_date) != 10:
        raise HTTPException(status_code=400, detail="game_date must be YYYY-MM-DD")
    try:
        sb = scoreboardv2.ScoreboardV2(
            game_date=game_date,
            timeout=60,
            headers=NBA_HEADERS,
        )
        return sb.get_dict()
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"
        raise HTTPException(status_code=502, detail=f"NBA stats fetch failed: {msg[:300]}")

@app.get("/player/{player_id}/gamelog")
def get_player_gamelog(player_id: int, season: str = DEFAULT_SEASON):
    df = gamelog_df(int(player_id), str(season))
    return {"player_id": int(player_id), "season": str(season), "rows": df.to_dict(orient="records")}

@app.get("/team/{team_id}/gamelog")
def get_team_gamelog(team_id: int, season: str = DEFAULT_SEASON, season_type: str = "Regular Season"):
    try:
        tg = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star=season_type,
            timeout=60,
            headers=NBA_HEADERS,
        )
        return tg.get_dict()
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"
        raise HTTPException(status_code=502, detail=f"NBA stats fetch failed: {msg[:300]}")

@app.get("/teamstats")
def get_team_stats(season: str = DEFAULT_SEASON, per_mode: str = "PerGame", season_type: str = "Regular Season"):
    try:
        ts = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed=per_mode,
            season_type_all_star=season_type,
            timeout=60,
            headers=NBA_HEADERS,
        )
        return ts.get_dict()
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"
        raise HTTPException(status_code=502, detail=f"NBA stats fetch failed: {msg[:300]}")

@app.get("/playerstats")
def get_player_stats(season: str = DEFAULT_SEASON, per_mode: str = "PerGame", season_type: str = "Regular Season"):
    try:
        ps = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed=per_mode,
            season_type_all_star=season_type,
            timeout=60,
            headers=NBA_HEADERS,
        )
        return ps.get_dict()
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"
        raise HTTPException(status_code=502, detail=f"NBA stats fetch failed: {msg[:300]}")

@app.get("/team/{team_id}/splits")
def get_team_splits(team_id: int, season: str = DEFAULT_SEASON, season_type: str = "Regular Season"):
    try:
        splits = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
            team_id=team_id,
            season=season,
            season_type_all_star=season_type,
            timeout=60,
            headers=NBA_HEADERS,
        )
        return splits.get_dict()
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"
        raise HTTPException(status_code=502, detail=f"NBA stats fetch failed: {msg[:300]}")

# ----------------------------
# Endpoint: prop prediction
# ----------------------------
@app.post("/predict/prop")
def predict_prop(payload: Dict[str, Any]):
    """
    payload example:
    {
      "player_id": 2544,
      "season": "2025-26",
      "stat": "PTS",
      "line": 27.5,
      "over_odds": -110,
      "under_odds": -110
    }
    """
    player_id = payload.get("player_id")
    season = payload.get("season", DEFAULT_SEASON)
    stat: str = payload.get("stat", "PTS")
    line = payload.get("line")

    over_odds = payload.get("over_odds")    # optional int
    under_odds = payload.get("under_odds")  # optional int

    if player_id is None:
        raise HTTPException(status_code=400, detail="player_id is required")
    if line is None:
        raise HTTPException(status_code=400, detail="line is required")
    if stat not in STAT_MAP:
        raise HTTPException(status_code=400, detail=f"stat must be one of {list(STAT_MAP.keys())}")

    df = gamelog_df(int(player_id), str(season))
    if df.empty:
        raise HTTPException(status_code=404, detail="No game log data returned (player/season may be wrong)")

    stat_col = STAT_MAP[stat]
    n_games = int(len(df))

    mins = estimate_minutes(df, last_n=10)
    rate = estimate_rate_per_minute(df, stat_col=stat_col, last_n=15)
    dist = combine_minutes_rate(mins, rate)

    mean = dist["mean"]
    sd = dist["sd"]

    p_over_model = prob_over(float(line), mean, sd)
    p_under_model = 1.0 - p_over_model

    p10 = normal_ppf_loc_scale(0.10, mean, sd)
    p50 = mean
    p90 = normal_ppf_loc_scale(0.90, mean, sd)

    confidence = confidence_label(mins["sd"], sd, n_games)

    market = None
    edge_pct = None
    stake = {"recommendation": "NO_BET", "units": 0.0}

    if isinstance(over_odds, int) and isinstance(under_odds, int):
        p_over_imp = american_to_implied_prob(over_odds)
        p_under_imp = american_to_implied_prob(under_odds)
        nv = no_vig_two_way(p_over_imp, p_under_imp)

        edge_pct = (p_over_model - nv["p_over"]) * 100.0
        market = {
            "over_odds": over_odds,
            "under_odds": under_odds,
            "implied": {"p_over": p_over_imp, "p_under": p_under_imp},
            "no_vig": nv,
        }
        stake = unit_sizing(edge_pct, confidence)

    pick = "NO_BET"
    if edge_pct is not None:
        if stake["recommendation"] == "PLAY":
            pick = "OVER" if p_over_model >= 0.5 else "UNDER"
    else:
        # no odds -> only lean if clearly away from 50%
        if confidence != "low" and abs(p_over_model - 0.5) >= 0.06:
            pick = "OVER" if p_over_model > 0.5 else "UNDER"

    return {
        "input": {
            "player_id": int(player_id),
            "season": str(season),
            "stat": stat,
            "line": float(line),
        },
        "projection": {
            "mean": round(mean, 2),
            "sd": round(sd, 2),
            "p10": round(p10, 2),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
        },
        "probabilities": {
            "p_over": round(p_over_model, 4),
            "p_under": round(p_under_model, 4),
        },
        "confidence": confidence,
        "market": market,
        "edge_pct_points": None if edge_pct is None else round(edge_pct, 2),
        "stake": stake,
        "recommendation": pick,
        "notes": {
            "n_games_used": n_games,
            "minutes_mean": round(mins["mean"], 2),
            "minutes_sd": round(mins["sd"], 2),
            "rate_per_min_mean": round(rate["mean"], 4),
            "rate_per_min_sd": round(rate["sd"], 4),
        },
    }
