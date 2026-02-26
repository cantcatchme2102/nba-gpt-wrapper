import os
from typing import Optional

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

app = FastAPI(title="NBA Stats Wrapper", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

DEFAULT_SEASON = os.getenv("DEFAULT_SEASON", "2025-26")


@app.get("/health")
def health():
    return {"ok": True, "default_season": DEFAULT_SEASON}


@app.get("/players/search")
def search_players(q: str):
    q = (q or "").strip()
    if len(q) < 2:
        raise HTTPException(status_code=400, detail="q must be at least 2 characters")
    results = players.find_players_by_full_name(q)
    return {"query": q, "results": results[:10]}


@app.get("/teams")
def list_teams():
    return {"results": teams.get_teams()}


@app.get("/scoreboard")
def get_scoreboard(game_date: str):
    game_date = (game_date or "").strip()
    if len(game_date) != 10:
        raise HTTPException(status_code=400, detail="game_date must be YYYY-MM-DD")
    sb = scoreboardv2.ScoreboardV2(game_date=game_date)
    return sb.get_dict()


@app.get("/player/{player_id}/gamelog")
def get_player_gamelog(player_id: int, season: str = DEFAULT_SEASON):
    gl = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    return gl.get_dict()


@app.get("/team/{team_id}/gamelog")
def get_team_gamelog(team_id: int, season: str = DEFAULT_SEASON, season_type: str = "Regular Season"):
    tg = teamgamelog.TeamGameLog(team_id=team_id, season=season, season_type_all_star=season_type)
    return tg.get_dict()


@app.get("/teamstats")
def get_team_stats(
    season: str = DEFAULT_SEASON,
    per_mode: str = "PerGame",
    season_type: str = "Regular Season",
):
    ts = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed=per_mode,
        season_type_all_star=season_type,
    )
    return ts.get_dict()


@app.get("/playerstats")
def get_player_stats(
    season: str = DEFAULT_SEASON,
    per_mode: str = "PerGame",
    season_type: str = "Regular Season",
):
    ps = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed=per_mode,
        season_type_all_star=season_type,
    )
    return ps.get_dict()


@app.get("/team/{team_id}/splits")
def get_team_splits(
    team_id: int,
    season: str = DEFAULT_SEASON,
    season_type: str = "Regular Season",
):
    splits = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
        team_id=team_id,
        season=season,
        season_type_all_star=season_type,
    )
    return splits.get_dict()


@app.get("/meta/team/by_abbr")
def team_by_abbr(abbr: str):
    abbr = (abbr or "").strip().upper()
    for t in teams.get_teams():
        if t.get("abbreviation") == abbr:
            return t
    raise HTTPException(status_code=404, detail=f"Team not found for abbreviation '{abbr}'")


@app.get("/meta/player/by_name")
def player_by_name(name: str):
    name = (name or "").strip()
    if len(name) < 2:
        raise HTTPException(status_code=400, detail="name must be at least 2 characters")
    results = players.find_players_by_full_name(name)
    if not results:
        raise HTTPException(status_code=404, detail=f"No players found for '{name}'")
    return {"best_match": results[0], "top_matches": results[:10]}
