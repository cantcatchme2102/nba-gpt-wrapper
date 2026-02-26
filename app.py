import os
import math
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog


DEFAULT_SEASON = os.getenv("DEFAULT_SEASON", "2025-26")

app = FastAPI(title="Local NBA Props API (Tunnel)", version="1.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ----------------------------
# NBA headers (important)
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

STAT_MAP = {"PTS": "PTS", "REB": "REB", "AST": "AST", "FG3M": "FG3M"}  # 3PM = FG3M


# ----------------------------
# Disk cache (survives restarts)
# ----------------------------
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CACHE_TTL_SECONDS = 60 * 60 * 6            # 6 hours = "fresh"
CACHE_STALE_MAX_SECONDS = 60 * 60 * 24 * 7 # 7 days max stale fallback


def _cache_path(player_id: int, season: str) -> Path:
    safe_season = season.replace("/", "-")
    return CACHE_DIR / f"gamelog_{player_id}_{safe_season}.json"


def _load_cache(player_id: int, season: str) -> Optional[Dict[str, Any]]:
    p = _cache_path(player_id, season)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cache(player_id: int, season: str, rows: List[Dict[str, Any]]) -> None:
    p = _cache_path(player_id, season)
    payload = {"ts": time.time(), "rows": rows}
    p.write_text(json.dumps(payload), encoding="utf-8")


def fetch_player_gamelog_rows(player_id: int, season: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (rows, meta)
      meta.source: live | cache_fresh | cache_stale_fallback
    """
    now = time.time()
    cached = _load_cache(player_id, season)

    # 1) Serve fresh cache
    if cached and isinstance(cached.get("rows"), list):
        age = now - float(cached.get("ts", 0))
        if age <= CACHE_TTL_SECONDS:
            return cached["rows"], {"source": "cache_fresh", "cache_age_seconds": int(age)}

    # 2) Try live pull ONCE (longer timeout to maximize chance of success)
    last_err = None
    try:
        gl = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            timeout=75,  # key change
            headers=NBA_HEADERS,
        )
        d = gl.get_dict()
        rs = d.get("resultSets", [])
        if not rs:
            rows = []
        else:
            headers = rs[0]["headers"]
            rowset = rs[0]["rowSet"]
            rows = [{headers[i]: r[i] for i in range(len(headers))} for r in rowset]

        _save_cache(player_id, season, rows)
        return rows, {"source": "live", "cache_age_seconds": 0}

    except Exception as e:
        last_err = e

    # 3) Fallback to stale cache (so GPT still answers)
    if cached and isinstance(cached.get("rows"), list):
        age = now - float(cached.get("ts", 0))
        if age <= CACHE_STALE_MAX_SECONDS:
            return cached["rows"], {
                "source": "cache_stale_fallback",
                "cache_age_seconds": int(age),
                "warning": "NBA live pull timed out; using cached data",
            }

    msg = f"{type(last_err).__name__}: {str(last_err)}"
    raise HTTPException(status_code=502, detail=f"NBA stats fetch failed (no usable cache): {msg[:300]}")


# ----------------------------
# Math helpers (pure python)
# ----------------------------
def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: List[float]) -> float:
    n = len(xs)
    if n <= 1:
        return 0.0
    m = mean(xs)
    var = sum((v - m) ** 2 for v in xs) / (n - 1)
    return math.sqrt(var)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    # Acklam approximation
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


def normal_ppf_loc_scale(p: float, mu: float, sd: float) -> float:
    return mu + sd * normal_ppf(p)


# ----------------------------
# Odds helpers
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
# Model: Minutes x Rate
# ----------------------------
def estimate_minutes(rows: List[Dict[str, Any]], last_n: int = 10) -> Dict[str, float]:
    mins = [safe_float(r.get("MIN", 0)) for r in rows[:last_n]]
    mins = [m for m in mins if m > 0]
    if not mins:
        return {"mean": 0.0, "sd": 8.0}
    mu = mean(mins)
    sd = stdev(mins) or 6.0
    return {"mean": mu, "sd": clamp(sd, 3.0, 12.0)}


def estimate_rate_per_minute(rows: List[Dict[str, Any]], stat_col: str, last_n: int = 15) -> Dict[str, float]:
    tail = rows[:last_n]
    rates: List[float] = []
    for r in tail:
        m = safe_float(r.get("MIN", 0))
        s = safe_float(r.get(stat_col, 0))
        if m > 0:
            rates.append(s / m)
    if not rates:
        return {"mean": 0.0, "sd": 0.05}
    mu = mean(rates)
    sd = stdev(rates) or max(0.03, 0.15 * mu)
    return {"mean": mu, "sd": clamp(sd, 0.02, 0.25)}


def combine_minutes_rate(mins: Dict[str, float], rate: Dict[str, float]) -> Dict[str, float]:
    m_mu, m_sd = mins["mean"], mins["sd"]
    r_mu, r_sd = rate["mean"], rate["sd"]
    mu = m_mu * r_mu
    var = (m_mu**2) * (r_sd**2) + (r_mu**2) * (m_sd**2) + (m_sd**2) * (r_sd**2)
    sd = math.sqrt(max(1e-9, var))
    return {"mean": mu, "sd": clamp(sd, 1.0, 15.0)}


def prob_over(line: float, mu: float, sd: float) -> float:
    if sd <= 0:
        return 1.0 if mu > line else 0.0
    return normal_cdf((mu - line) / sd)


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
    if edge_pct < 2.0 or confidence == "low":
        return {"recommendation": "NO_BET", "units": 0.0}
    if edge_pct < 3.5:
        return {"recommendation": "PLAY", "units": 0.5}
    if edge_pct < 6.0:
        return {"recommendation": "PLAY", "units": 1.0}
    return {"recommendation": "PLAY", "units": 1.5}


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "message": "NBA API running. Try /health, /players/search, /meta/player/by_name, /warmup/player/{id}, /predict/prop",
    }


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
    res = players.find_players_by_full_name(name)
    if not res:
        raise HTTPException(status_code=404, detail=f"No players found for '{name}'")
    return {"best_match": res[0], "top_matches": res[:10]}


@app.get("/player/{player_id}/gamelog")
def get_player_gamelog(player_id: int, season: str = DEFAULT_SEASON):
    rows, meta = fetch_player_gamelog_rows(int(player_id), str(season))
    return {"player_id": int(player_id), "season": str(season), "meta": meta, "rows": rows}


@app.get("/warmup/player/{player_id}")
def warmup_player(player_id: int, season: str = DEFAULT_SEASON):
    rows, meta = fetch_player_gamelog_rows(int(player_id), str(season))
    return {
        "ok": True,
        "player_id": int(player_id),
        "season": str(season),
        "rows_cached": len(rows),
        "meta": meta,
    }


@app.post("/predict/prop")
def predict_prop(payload: Dict[str, Any]):
    player_id = payload.get("player_id")
    season = payload.get("season", DEFAULT_SEASON)
    stat = payload.get("stat", "PTS")
    line = payload.get("line")

    over_odds = payload.get("over_odds")    # optional int
    under_odds = payload.get("under_odds")  # optional int

    if player_id is None:
        raise HTTPException(status_code=400, detail="player_id is required")
    if line is None:
        raise HTTPException(status_code=400, detail="line is required")
    if stat not in STAT_MAP:
        raise HTTPException(status_code=400, detail=f"stat must be one of {list(STAT_MAP.keys())}")

    rows, data_meta = fetch_player_gamelog_rows(int(player_id), str(season))
    if not rows:
        raise HTTPException(status_code=404, detail="No game log rows returned")

    n_games = len(rows)
    stat_col = STAT_MAP[stat]

    mins = estimate_minutes(rows, last_n=10)
    rate = estimate_rate_per_minute(rows, stat_col=stat_col, last_n=15)
    dist = combine_minutes_rate(mins, rate)

    mu = dist["mean"]
    sd = dist["sd"]

    p_over = prob_over(float(line), mu, sd)
    p_under = 1.0 - p_over

    p10 = normal_ppf_loc_scale(0.10, mu, sd)
    p50 = mu
    p90 = normal_ppf_loc_scale(0.90, mu, sd)

    conf = confidence_label(mins["sd"], sd, n_games)

    market = None
    edge_pct = None
    stake = {"recommendation": "NO_BET", "units": 0.0}

    if isinstance(over_odds, int) and isinstance(under_odds, int):
        p_over_imp = american_to_implied_prob(over_odds)
        p_under_imp = american_to_implied_prob(under_odds)
        nv = no_vig_two_way(p_over_imp, p_under_imp)

        edge_pct = (p_over - nv["p_over"]) * 100.0
        market = {
            "over_odds": over_odds,
            "under_odds": under_odds,
            "implied": {"p_over": p_over_imp, "p_under": p_under_imp},
            "no_vig": nv,
        }
        stake = unit_sizing(edge_pct, conf)

    pick = "NO_BET"
    if edge_pct is not None and stake["recommendation"] == "PLAY":
        pick = "OVER" if p_over >= 0.5 else "UNDER"
    elif edge_pct is None:
        if conf != "low" and abs(p_over - 0.5) >= 0.06:
            pick = "OVER" if p_over > 0.5 else "UNDER"

    return {
        "input": {"player_id": int(player_id), "season": str(season), "stat": stat, "line": float(line)},
        "data_meta": data_meta,
        "projection": {
            "mean": round(mu, 2),
            "sd": round(sd, 2),
            "p10": round(p10, 2),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
        },
        "probabilities": {"p_over": round(p_over, 4), "p_under": round(p_under, 4)},
        "confidence": conf,
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
            "cache_dir": str(CACHE_DIR),
        },
    }
