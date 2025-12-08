# app/services/trade_pmf_eval.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional

from espn_api.basketball import League

from app.services.PMF_utils import (
    build_team_pmf_counting,
    build_team_pmf_2d,
    calculate_win_probability,
    calculate_percentage_win_probability,
    load_player_pmfs,
    compress_pmf,
    compress_ratio_pmf_from_2d,
    expected_ratio_from_2d_pmf,
    PMF_CACHE_DIR,
)

# ---------- Category definitions ----------

ALL_CATEGORIES: List[str] = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "FG%", "FT%", "TO"]

CATEGORY_COLUMN_MAP: Dict[str, str] = {
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "STL": "STL",
    "BLK": "BLK",
    "3PM": "FG3M",
    "TO":  "TOV",
}

PERCENTAGE_CATEGORIES: Dict[str, Tuple[str, str]] = {
    "FG%": ("FGM", "FGA"),
    "FT%": ("FTM", "FTA"),
}

# =========================
#   Roster / Trade setup
# =========================

def _build_team_players_from_rosters(
    league: League,
    games_per_player: int = 3,
    allowed_player_names: Optional[Set[str]] = None,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, int]]:
    """
    Build team->players map from ESPN rosters.
    Hardcodes games to `games_per_player` (default 3) and assumes ACTIVE status.
    If allowed_player_names is provided, only include players in that set.
    """
    team_players: Dict[int, List[Dict[str, Any]]] = {}
    player_to_team: Dict[str, int] = {}

    for team_idx, team in enumerate(league.teams):
        plist: List[Dict[str, Any]] = []
        for p in team.roster:
            name = str(getattr(p, "name", "")) or ""
            if not name:
                continue

            # Filter if set is provided
            if allowed_player_names is not None and name not in allowed_player_names:
                continue

            # Simplified: Hardcode 3 games, assume ACTIVE
            rec = {
                "player_name": name,
                "games": games_per_player,
                "inj": "ACTIVE", 
            }
            plist.append(rec)

            if name not in player_to_team:
                player_to_team[name] = team_idx

        team_players[team_idx] = plist

    return team_players, player_to_team


def _simulate_trade_on_rosters(
    team_players_before: Dict[int, List[Dict[str, Any]]],
    player_to_team: Dict[str, int],
    side_a: List[str],
    side_b: List[str],
    league: League,
    forced_team_a_idx: Optional[int] = None,
    forced_team_b_idx: Optional[int] = None,
) -> Tuple[Optional[int], Optional[int], Dict[int, List[Dict[str, Any]]]]:
    """
    Apply the proposed trade to the team_players_before mapping.
    """
    traded_a = [n for n in side_a if n]
    traded_b = [n for n in side_b if n]

    # Identify teams if not forced
    teams_in_a = {player_to_team[n] for n in traded_a if n in player_to_team}
    teams_in_b = {player_to_team[n] for n in traded_b if n in player_to_team}

    team_a_idx = forced_team_a_idx
    team_b_idx = forced_team_b_idx

    # Heuristic: If we can't find the teams from the inputs, we can't proceed.
    # But with "Side A = Destination A", we need to know who Team A is.
    
    if team_a_idx is None and teams_in_b:
        # Side B players are going TO Team B. They are currently on Team A (usually).
        # So teams_in_b likely contains Team A.
        team_a_idx = sorted(teams_in_b)[0]
        
    if team_b_idx is None and teams_in_a:
        # Side A players are going TO Team A. They are currently on Team B.
        # So teams_in_a likely contains Team B.
        team_b_idx = sorted(teams_in_a)[0]

    if team_a_idx is None or team_b_idx is None:
        print("[TRADE-PMF] Could not identify both trading teams.")
        return None, None, team_players_before
    
    if team_a_idx == team_b_idx:
        print("[TRADE-PMF] Same team identified for both sides.")
        return None, None, team_players_before

    # Deep copy rosters
    team_players_after: Dict[int, List[Dict[str, Any]]] = {
        idx: [dict(p) for p in players]
        for idx, players in team_players_before.items()
    }

    def _move_player(player_name: str, dst_idx: int) -> None:
        # Find player's current team
        src_idx = player_to_team.get(player_name)
        if src_idx is None:
            print(f"[TRADE-PMF] WARN: {player_name} not found in league.")
            return
        
        if src_idx == dst_idx:
            # Already on target team
            return

        src_list = team_players_after.get(src_idx, [])
        found = False
        for i, p in enumerate(src_list):
            if p.get("player_name") == player_name:
                src_list.pop(i)
                team_players_after.setdefault(dst_idx, []).append(p)
                found = True
                break
        
        if not found:
            # Player might have moved already if listed twice?
            pass

    # STRICT LOGIC: Side A players go TO Team A
    for name in traded_a:
        _move_player(name, team_a_idx)

    # STRICT LOGIC: Side B players go TO Team B
    for name in traded_b:
        _move_player(name, team_b_idx)

    return team_a_idx, team_b_idx, team_players_after

    return team_a_idx, team_b_idx, team_players_after


# =========================
#   PMF building helpers
# =========================

def _build_all_team_pmfs(
    team_players_map: Dict[int, List[Dict[str, Any]]],
    season: str,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Build 1D and 2D PMFs for all teams.
    """
    pmf_1d: Dict[int, Dict[str, Any]] = {}
    pmf_2d: Dict[int, Dict[str, Any]] = {}

    for t_idx, players in team_players_map.items():
        pmf_1d[t_idx] = {}
        pmf_2d[t_idx] = {}

        for cat in ALL_CATEGORIES:
            if cat in ("FG%", "FT%"):
                makes_col, attempts_col = PERCENTAGE_CATEGORIES[cat]
                pmf_2d[t_idx][cat] = build_team_pmf_2d(
                    players,
                    makes_col=makes_col,
                    attempts_col=attempts_col,
                    season=season,
                    load_player_pmfs=load_player_pmfs,
                )
            else:
                stat_col = CATEGORY_COLUMN_MAP.get(cat)
                pmf_1d[t_idx][cat] = build_team_pmf_counting(
                    players,
                    stat_col=stat_col,
                    season=season,
                    load_player_pmfs=load_player_pmfs,
                )

    return pmf_1d, pmf_2d


def _avg_win_pct_for_team(
    team_idx: int,
    league: League,
    pmf1_map: Dict[int, Dict[str, Any]],
    pmf2_map: Dict[int, Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute average win% vs all other teams per category.
    """
    res: Dict[str, float] = {}
    num_teams = len(league.teams)

    for cat in ALL_CATEGORIES:
        total_prob = 0.0
        count = 0

        for opp_idx in range(num_teams):
            if opp_idx == team_idx:
                continue

            if cat in ("FG%", "FT%"):
                t_pmf = pmf2_map[team_idx][cat]
                o_pmf = pmf2_map[opp_idx][cat]
                p = calculate_percentage_win_probability(t_pmf, o_pmf)
            else:
                t_pmf = pmf1_map[team_idx][cat]
                o_pmf = pmf1_map[opp_idx][cat]

                if cat == "TO":
                    # lower TO is better
                    # calculate_win_probability(A, B) = P(A > B)
                    # We want P(Team < Opp) = P(Opp > Team)
                    p = calculate_win_probability(o_pmf, t_pmf)
                else:
                    p = calculate_win_probability(t_pmf, o_pmf)

            total_prob += p
            count += 1

        avg_pct = (total_prob / count) * 100.0 if count else 50.0
        res[cat] = avg_pct

    return res


# =========================
#   Main entry point
# =========================

def evaluate_trade_with_pmfs(
    league: League,
    year: int,
    stat_window: str,
    side_a: List[str],
    side_b: List[str],
    team_a_idx: Optional[int] = None,  # ESPN team_id from UI
    team_b_idx: Optional[int] = None,  # ESPN team_id from UI
    allowed_player_names: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Simplified PMF-based trade assessment.
    """
    print(f"[TRADE-PMF] Evaluating trade: A={side_a} B={side_b}")

    # Map ESPN team_id to internal index
    idx_to_id: Dict[int, int] = {}
    id_to_idx: Dict[int, int] = {}
    for idx, t in enumerate(league.teams):
        tid = getattr(t, "team_id", idx + 1)
        idx_to_id[idx] = tid
        id_to_idx[tid] = idx

    forced_team_a_idx = id_to_idx.get(team_a_idx) if team_a_idx is not None else None
    forced_team_b_idx = id_to_idx.get(team_b_idx) if team_b_idx is not None else None

    # 1. Build base rosters (Hardcoded 3 games)
    team_players_before, player_to_team = _build_team_players_from_rosters(
        league,
        games_per_player=3,
        allowed_player_names=allowed_player_names,
    )

    # 2. Simulate Trade
    team_a_idx_int, team_b_idx_int, team_players_after = _simulate_trade_on_rosters(
        team_players_before=team_players_before,
        player_to_team=player_to_team,
        side_a=side_a,
        side_b=side_b,
        league=league,
        forced_team_a_idx=forced_team_a_idx,
        forced_team_b_idx=forced_team_b_idx,
    )

    if team_a_idx_int is None or team_b_idx_int is None:
        return None

    team_a_id = idx_to_id[team_a_idx_int]
    team_b_id = idx_to_id[team_b_idx_int]
    season_str = str(year)

    # 3. Build PMFs
    print("[TRADE-PMF] Building 'Before' PMFs...")
    pmf1_before, pmf2_before = _build_all_team_pmfs(team_players_before, season_str)
    
    print("[TRADE-PMF] Building 'After' PMFs...")
    pmf1_after, pmf2_after = _build_all_team_pmfs(team_players_after, season_str)

    # 4. Calculate Win %
    print("[TRADE-PMF] Calculating Win %...")
    before_a = _avg_win_pct_for_team(team_a_idx_int, league, pmf1_before, pmf2_before)
    after_a  = _avg_win_pct_for_team(team_a_idx_int, league, pmf1_after,  pmf2_after)

    before_b = _avg_win_pct_for_team(team_b_idx_int, league, pmf1_before, pmf2_before)
    after_b  = _avg_win_pct_for_team(team_b_idx_int, league, pmf1_after,  pmf2_after)

    # 4b. Calculate Raw Stats (Means)
    print("[TRADE-PMF] Calculating Raw Stats...")
    
    def _calc_stats(team_idx, pmf1_map, pmf2_map):
        stats = {}
        for cat in ALL_CATEGORIES:
            if cat in ("FG%", "FT%"):
                # Expected ratio * 100 for percentage
                pmf = pmf2_map[team_idx][cat]
                val = expected_ratio_from_2d_pmf(pmf) * 100.0
                print(f"[DEBUG] {cat} for team {team_idx}: ratio={val/100.0:.4f}, val={val:.4f}")
            else:
                # Mean of 1D PMF
                pmf = pmf1_map[team_idx][cat]
                val = pmf.mean()
            stats[cat] = val
        return stats

    stats_before_a = _calc_stats(team_a_idx_int, pmf1_before, pmf2_before)
    stats_after_a  = _calc_stats(team_a_idx_int, pmf1_after,  pmf2_after)
    stats_before_b = _calc_stats(team_b_idx_int, pmf1_before, pmf2_before)
    stats_after_b  = _calc_stats(team_b_idx_int, pmf1_after,  pmf2_after)

    # 5. Compress PMFs for Frontend
    def _compress_all(pmf1_map, pmf2_map):
        c_1d = {}
        c_2d = {}
        num_teams = len(league.teams)
        
        for cat in ALL_CATEGORIES:
            if cat in ("FG%", "FT%"):
                c_2d[cat] = []
                for idx in range(num_teams):
                    t = league.teams[idx]
                    tid = idx_to_id[idx]
                    pmf = pmf2_map[idx][cat]
                    c_2d[cat].append({
                        "team_idx": tid,
                        "team_name": t.team_name,
                        "pmf": compress_ratio_pmf_from_2d(pmf),
                    })
            else:
                c_1d[cat] = []
                for idx in range(num_teams):
                    t = league.teams[idx]
                    tid = idx_to_id[idx]
                    pmf = pmf1_map[idx][cat]
                    c_1d[cat].append({
                        "team_idx": tid,
                        "team_name": t.team_name,
                        "pmf": compress_pmf(pmf),
                    })
        return c_1d, c_2d

    before_1d, before_2d = _compress_all(pmf1_before, pmf2_before)
    after_1d, after_2d = _compress_all(pmf1_after, pmf2_after)

    # 6. Format Results
    result = {
        "teams": [
            {"team_idx": idx_to_id[i], "team_name": t.team_name, "is_trading": i in (team_a_idx_int, team_b_idx_int)} 
            for i, t in enumerate(league.teams)
        ],
        "categories": ALL_CATEGORIES,
        "trading_teams": {
            "team_a_idx": team_a_id,
            "team_b_idx": team_b_id,
            "team_a_name": league.teams[team_a_idx_int].team_name,
            "team_b_name": league.teams[team_b_idx_int].team_name,
        },
        "before": {
            "avg_win_pct": {
                team_a_id: before_a,
                team_b_id: before_b,
            },
            "avg_stats": {
                team_a_id: stats_before_a,
                team_b_id: stats_before_b,
            },
            "pmfs": {
                "1d": before_1d,
                "2d": before_2d,
            }
        },
        "after": {
            "avg_win_pct": {
                team_a_id: after_a,
                team_b_id: after_b,
            },
            "avg_stats": {
                team_a_id: stats_after_a,
                team_b_id: stats_after_b,
            },
            "pmfs": {
                "1d": after_1d,
                "2d": after_2d,
            }
        }
    }

    return result
