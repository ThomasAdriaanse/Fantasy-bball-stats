# app/services/compare_presenter.py
from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any, Optional
import math
import os

# ---------- Tunables ----------

# Snapshot intensity baselines (drives split-pill background alpha)
_SCALE_MAP: Dict[str, float] = {
    'PTS': 100,
    'REB': 30,
    'AST': 20,
    'STL': 15,
    'BLK': 10,
    '3PM': 15,
    'FG%': 0.50,   # proportions (0..1) expected in inputs
    'FT%': 0.50,   # proportions (0..1) expected in inputs
    'TO' : 10,
}

_DISPLAY_ORDER: List[str] = ['PTS','REB','AST','STL','BLK','3PM','FG%','FT%','TO']

# Map raw keys → display label for odds (kept for compatibility)
_KEY_MAP_ODDS: Dict[str, str] = {
    "threeptm": "3PM",
    "fg3m": "3PM",
    "fg3_pm": "3PM",
    "3ptm": "3PM",
    "3pm": "3PM",
    "fg%": "FG%",
    "ft%": "FT%",
    "turno": "TO",
    "to": "TO",
}

# ---------- Helpers ----------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "-"

def _latest_predicted(combined_jsons: Dict[str, List[Dict[str, Any]]],
                      cat: str,
                      team_label: str) -> Optional[float]:
    """
    Pull the latest predicted value for a category & team from combined_jsons.
    For FG%/FT% this is typically a proportion (0..1); for counting cats a count.
    (Kept for compatibility; not used by the v2 odds path.)
    """
    series = combined_jsons.get(cat) or []
    if not series:
        return None
    try:
        rows = [r for r in series if r.get("team") == team_label]
        rows.sort(key=lambda r: datetime.strptime(str(r.get("date")), "%Y-%m-%d"))
        if not rows:
            return None
        return float(rows[-1].get("predicted_cat"))
    except Exception:
        return None

def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _num(x, default=0.0) -> float:
    try:
        if isinstance(x, dict):
            # Support {'value': ...}
            return float(x.get('value', default))
        return float(x)
    except Exception:
        return float(default)

def _get_cur_total(cur: Dict[str, Any], *keys: str) -> float:
    """
    Try several keys (case-insensitive) to fetch a numeric current total.
    Supports both flat floats and {'value': float}.
    """
    if not cur:
        return 0.0
    for k in keys:
        if k in cur:
            return _num(cur[k], 0.0)
        kl = k.lower(); ku = k.upper(); kt = k.title()
        for c in (kl, ku, kt):
            if c in cur:
                return _num(cur[c], 0.0)
        s = cur.get(k) or cur.get(ku) or cur.get(kl) or cur.get(kt)
        if isinstance(s, dict) and 'value' in s:
            return _num(s['value'], 0.0)
    return 0.0

# --- Availability helpers (OUT players shouldn't count) ---

def _is_out(status: Optional[str]) -> bool:
    if status is None:
        return False
    s = str(status).strip().upper()
    return s in {"OUT", "INJURY_RESERVE", "IR", "SUSPENDED"}

def _filter_active_players(rows: List[Dict[str, Any]],
                           exclude_day_to_day: bool = False) -> List[Dict[str, Any]]:
    """Keep players who are not OUT and who still have >0 remaining games."""
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        inj = r.get("inj")
        if _is_out(inj):
            continue
        if exclude_day_to_day and str(inj or "").strip().upper() == "DAY_TO_DAY":
            continue
        g = r.get("games")
        try:
            g = float(g)
        except Exception:
            g = 0.0
        if g <= 0:
            continue
        out.append(r)
    return out

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _project_counting(players: List[Dict[str, Any]],
                      col: str,
                      games_col: str = 'games',
                      *,
                      exclude_day_to_day: bool = False) -> float:
    """
    Sum of per-player (per-game average) * remaining games for a counting category.
    Excludes OUT players (and zero-game rows). Optionally excludes DAY_TO_DAY.
    """
    total = 0.0
    for r in _filter_active_players(players, exclude_day_to_day=exclude_day_to_day):
        g = _safe_float(r.get(games_col), 0.0)
        v = _safe_float(r.get(col), 0.0)
        total += v * g
    return float(total)

def _project_makes_attempts(players: List[Dict[str, Any]],
                            made_col: str,
                            att_col: str,
                            games_col: str = 'games',
                            *,
                            exclude_day_to_day: bool = False) -> tuple[float, float]:
    """
    Sum of per-player (per-game) makes/attempts * remaining games for % categories.
    Excludes OUT players (and zero-game rows). Optionally excludes DAY_TO_DAY.
    """
    tm = ta = 0.0
    for r in _filter_active_players(players, exclude_day_to_day=exclude_day_to_day):
        g = _safe_float(r.get(games_col), 0.0)
        m = _safe_float(r.get(made_col), 0.0)
        a = _safe_float(r.get(att_col), 0.0)
        tm += m * g
        ta += a * g
    return float(tm), float(ta)

def _label_class(p_left: float, p_right: float) -> str:
    if abs(p_left - p_right) < 0.5:
        return "is-tie"
    return "winner-left" if p_left > p_right else "winner-right"

# ---------- Snapshot (unchanged visual logic) ----------

def build_snapshot_rows(team1_current_stats: Optional[Dict[str, Any]],
                        team2_current_stats: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build split-pill data (values + alpha intensities) for the snapshot bar grid.
    """
    out: List[Dict[str, Any]] = []
    t1 = team1_current_stats or {}
    t2 = team2_current_stats or {}

    for cat in _DISPLAY_ORDER:
        s1 = t1.get(cat) if isinstance(t1, dict) else None
        s2 = t2.get(cat) if isinstance(t2, dict) else None
        v1 = (s1 or {}).get("value") if isinstance(s1, dict) else None
        v2 = (s2 or {}).get("value") if isinstance(s2, dict) else None

        a1 = a2 = 0.0
        disp1 = "-"
        disp2 = "-"

        if v1 is not None and v2 is not None:
            try:
                v1f = float(v1)
                v2f = float(v2)
                diff = abs(v1f - v2f)
                scale = float(_SCALE_MAP.get(cat, 10.0))
                intensity = _clamp(diff / scale, 0.0, 1.0)
                alpha = 0.25 + 0.55 * intensity

                if cat == 'TO':  # lower is better
                    if v1f < v2f: a1 = alpha
                    elif v2f < v1f: a2 = alpha
                else:
                    if v1f > v2f: a1 = alpha
                    elif v2f > v1f: a2 = alpha

                if cat in ('FG%','FT%'):
                    disp1 = _fmt_pct(v1f)
                    disp2 = _fmt_pct(v2f)
                else:
                    disp1 = v1 if v1 is not None else "-"
                    disp2 = v2 if v2 is not None else "-"
            except Exception:
                pass
        else:
            if cat in ('FG%','FT%'):
                disp1 = _fmt_pct(v1 if v1 is not None else None)
                disp2 = _fmt_pct(v2 if v2 is not None else None)

        out.append({
            "cat": cat,
            "v1": v1, "v2": v2,
            "disp1": disp1, "disp2": disp2,
            "a1": round(a1, 3), "a2": round(a2, 3),
        })

    return out

# ---------- Odds (v2 spec + OUT player filtering + deterministic no-games-left) ----------

_DEBUG_DEFAULT = os.getenv("COMPARE_DEBUG", "0").lower() in ("1", "true", "yes", "y")

def _dbg(enabled: bool, *args):
    if enabled:
        print("[compare_presenter][build_odds_rows]", *args, flush=True)

def build_odds_rows(
    win1_list: List[Dict[str, Any]],               # kept for compatibility (unused in v2 math)
    combined_jsons: Dict[str, List[Dict[str, Any]]],  # kept for compatibility (unused in v2 math)
    expected_pct_map: Optional[Dict[str, Dict[str, float]]] = None,  # kept for compatibility
    *,
    team1_current_stats: Optional[Dict[str, Any]] = None,
    team2_current_stats: Optional[Dict[str, Any]] = None,
    data_team_players_1: Optional[List[Dict[str, Any]]] = None,
    data_team_players_2: Optional[List[Dict[str, Any]]] = None,
    exclude_day_to_day: bool = False,
    debug: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Implements the 6-step spec, with deterministic handling when **no games remain**:

    • Counting cats: if proj1 == proj2 == 0 → compare current totals → 100/0 (or 50/50).
    • FG%/FT%: if proj_a1 == proj_a2 == 0 → compare current % (cur_m/cur_a). If both have
      no current attempts → 50/50.

    OUT players (and zero-game rows) are excluded from projections. Optionally drop DAY_TO_DAY.
    """
    use_debug = _DEBUG_DEFAULT if debug is None else bool(debug)
    out: List[Dict[str, Any]] = []

    if not (team1_current_stats and team2_current_stats and
            data_team_players_1 is not None and data_team_players_2 is not None):
        _dbg(use_debug,
             "Missing v2 inputs; need team1_current_stats, team2_current_stats, "
             "data_team_players_1, data_team_players_2. Returning [].")
        return out

    def dbg(*args):
        if use_debug: print("[odds_v2]", *args, flush=True)

    COUNT_MAP = {
        'PTS': 'pts',
        'REB': 'reb',
        'AST': 'ast',
        'STL': 'stl',
        'BLK': 'blk',
        '3PM': 'threeptm',
        'TO' : 'turno',
    }
    PCT_MAP = {
        'FG%': ('fgm', 'fga'),
        'FT%': ('ftm', 'fta'),
    }

    EPS = 1e-9

    for cat in _DISPLAY_ORDER:
        dbg(f"=== {cat} ===")

        if cat in COUNT_MAP:
            col = COUNT_MAP[cat]

            # Step 1: current totals
            cur1 = _get_cur_total(team1_current_stats, cat, col)
            cur2 = _get_cur_total(team2_current_stats, cat, col)

            # Step 2-3: projected remaining (with OUT/zero-game filtering)
            proj1 = _project_counting(data_team_players_1, col, exclude_day_to_day=exclude_day_to_day)
            proj2 = _project_counting(data_team_players_2, col, exclude_day_to_day=exclude_day_to_day)

            # Deterministic path when no games remain for either team
            if abs(proj1) < EPS and abs(proj2) < EPS:
                mu1 = cur1
                mu2 = cur2
                def _fmt(x: float) -> str:
                    return f"{x:.1f}" if abs(x) < 5 else f"{round(x)}"
                mid_t1 = _fmt(mu1)
                mid_t2 = _fmt(mu2)
                if abs(mu1 - mu2) < EPS:
                    p_left, p_right = 50.0, 50.0
                else:
                    # TO: lower better; other cats: higher better
                    t1_wins = (mu1 > mu2) if cat != 'TO' else (mu1 < mu2)
                    p_left, p_right = (100.0, 0.0) if t1_wins else (0.0, 100.0)
                dbg(f"[COUNT|DETERMINISTIC] no projections; cur1={cur1}, cur2={cur2} → {p_left}/{p_right}")
                out.append({
                    "cat": cat,
                    "p1": p_left, "p2": p_right,
                    "class_name": _label_class(p_left, p_right),
                    "mid_t1": mid_t1,
                    "mid_t2": mid_t2,
                })
                continue

            # Probabilistic path
            mu1 = cur1 + proj1
            mu2 = cur2 + proj2
            var1 = proj1 if proj1 > 0 else 1.0
            var2 = proj2 if proj2 > 0 else 1.0
            diff = (mu1 - mu2) if cat != 'TO' else (mu2 - mu1)  # TO lower is better
            denom = math.sqrt(max(var1 + var2, EPS))
            z = diff / denom
            p_team1 = _normal_cdf(z) * 100.0

            def _fmt(x: float) -> str:
                return f"{x:.1f}" if abs(x) < 5 else f"{round(x)}"
            mid_t1 = _fmt(mu1)
            mid_t2 = _fmt(mu2)
            p_left = round(p_team1, 1)
            p_right = round(100.0 - p_team1, 1)
            dbg(f"[COUNT] cur1={cur1}, cur2={cur2}, proj1={proj1}, proj2={proj2}, "
                f"mu=({mu1},{mu2}), var=({var1},{var2}), z={z:.4f}, "
                f"p_left={p_left}, p_right={p_right}, labels=({mid_t1},{mid_t2})")

            out.append({
                "cat": cat,
                "p1": p_left, "p2": p_right,
                "class_name": _label_class(p_left, p_right),
                "mid_t1": mid_t1,
                "mid_t2": mid_t2,
            })

        elif cat in PCT_MAP:
            made_col, att_col = PCT_MAP[cat]

            # Step 1: current totals (makes/attempts)
            cur_m1 = _get_cur_total(team1_current_stats, made_col, made_col.upper(), 'made_' + made_col)
            cur_a1 = _get_cur_total(team1_current_stats, att_col, att_col.upper(), 'att_' + att_col)
            cur_m2 = _get_cur_total(team2_current_stats, made_col, made_col.upper(), 'made_' + made_col)
            cur_a2 = _get_cur_total(team2_current_stats, att_col, att_col.upper(), 'att_' + att_col)

            # Step 2-3: projected remaining makes/attempts (active only)
            proj_m1, proj_a1 = _project_makes_attempts(
                data_team_players_1, made_col, att_col, exclude_day_to_day=exclude_day_to_day
            )
            proj_m2, proj_a2 = _project_makes_attempts(
                data_team_players_2, made_col, att_col, exclude_day_to_day=exclude_day_to_day
            )

            # Deterministic path when no projected attempts for either team
            if abs(proj_a1) < EPS and abs(proj_a2) < EPS:
                # compare current percentages (if any attempts)
                p1_cur = (cur_m1 / cur_a1) if cur_a1 > 0 else None
                p2_cur = (cur_m2 / cur_a2) if cur_a2 > 0 else None

                mid_t1 = "-" if p1_cur is None else f"{p1_cur * 100.0:.1f}%"
                mid_t2 = "-" if p2_cur is None else f"{p2_cur * 100.0:.1f}%"

                if (p1_cur is None) and (p2_cur is None):
                    p_left, p_right = 50.0, 50.0
                elif p1_cur is None:
                    p_left, p_right = 0.0, 100.0
                elif p2_cur is None:
                    p_left, p_right = 100.0, 0.0
                else:
                    if abs(p1_cur - p2_cur) < EPS:
                        p_left, p_right = 50.0, 50.0
                    else:
                        p_left, p_right = (100.0, 0.0) if p1_cur > p2_cur else (0.0, 100.0)

                dbg(f"[PCT|DETERMINISTIC] no projected attempts; "
                    f"cur p1={p1_cur}, p2={p2_cur} → {p_left}/{p_right}")
                out.append({
                    "cat": cat,
                    "p1": p_left, "p2": p_right,
                    "class_name": _label_class(p_left, p_right),
                    "mid_t1": mid_t1,
                    "mid_t2": mid_t2,
                })
                continue

            # Probabilistic path
            tot_m1 = cur_m1 + proj_m1; tot_a1 = cur_a1 + proj_a1
            tot_m2 = cur_m2 + proj_m2; tot_a2 = cur_a2 + proj_a2
            p1 = (tot_m1 / tot_a1) if tot_a1 > 0 else None
            p2 = (tot_m2 / tot_a2) if tot_a2 > 0 else None

            mid_t1 = "-" if p1 is None else f"{p1 * 100.0:.1f}%"
            mid_t2 = "-" if p2 is None else f"{p2 * 100.0:.1f}%"

            if (p1 is None) and (p2 is None):
                p_team1 = 50.0
                dbg(f"[PCT] no attempts both sides; coin flip.")
            elif p1 is None:
                p_team1 = 0.0
                dbg(f"[PCT] team1 has no attempts; P(win)=0.")
            elif p2 is None:
                p_team1 = 100.0
                dbg(f"[PCT] team2 has no attempts; P(win)=100.")
            else:
                var_diff = p1 * (1.0 - p1) / max(tot_a1, EPS) + p2 * (1.0 - p2) / max(tot_a2, EPS)
                z = (p1 - p2) / math.sqrt(max(var_diff, EPS))
                p_team1 = _normal_cdf(z) * 100.0
                dbg(f"[PCT] tot (m/a) t1=({tot_m1:.2f}/{tot_a1:.2f}), t2=({tot_m2:.2f}/{tot_a2:.2f}); "
                    f"p1={p1:.4f}, p2={p2:.4f}, var_diff={var_diff:.6e}, z={z:.4f}, "
                    f"P(win)={p_team1:.2f}")

            p_left = round(p_team1, 1)
            p_right = round(100.0 - p_team1, 1)

            out.append({
                "cat": cat,
                "p1": p_left, "p2": p_right,
                "class_name": _label_class(p_left, p_right),
                "mid_t1": mid_t1,
                "mid_t2": mid_t2,
            })

        else:
            out.append({
                "cat": cat, "p1": 50.0, "p2": 50.0,
                "class_name": "is-tie",
                "mid_t1": "-", "mid_t2": "-"
            })
            dbg(f"[WARN] Unhandled category label: {cat}")

    return out
