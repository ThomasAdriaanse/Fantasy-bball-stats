
from flask import Blueprint, render_template
from app.services import darko_services

bp = Blueprint('rankings', __name__)

@bp.route('/rankings')
def index():
    # Get raw data with Z-scores
    darko_data = darko_services.get_darko_z_scores()

    # Calculate Total Z and sort
    # item structure: { "player_name": ..., "team": ..., "RAW_DARKO": ..., "Z_DARKO": {...}, "RAW_REAL": ..., "Z_REAL": {...} }
    
    ranked_players = []
    for p in darko_data:
        # 1. Total DARKO Z
        z_scores_d = p.get("Z_DARKO", {})
        total_d = 0.0
        for k, v in z_scores_d.items():
            try:
                total_d += float(v)
            except (ValueError, TypeError):
                pass
        p["total_z_darko"] = total_d
        
        # 2. Total Real Z
        z_scores_r = p.get("Z_REAL", {})
        total_r = 0.0
        # Check if z_scores_r is empty (missing data)
        has_real_data = bool(z_scores_r)
        
        if has_real_data:
            for k, v in z_scores_r.items():
                try:
                    total_r += float(v)
                except (ValueError, TypeError):
                    pass
        p["total_z_real"] = total_r
        p["has_real_data"] = has_real_data

        # 3. Difference (DARKO - Real)
        p["Z_DIFF"] = {}
        if has_real_data:
            p["diff_z"] = total_d - total_r
            # Calculate per-category diff using the keys from Z_DARKO
            for k, v in z_scores_d.items():
                real_val = z_scores_r.get(k)
                if real_val is not None:
                    try:
                        p["Z_DIFF"][k] = float(v) - float(real_val)
                    except (ValueError, TypeError):
                        p["Z_DIFF"][k] = 0.0
                else:
                    p["Z_DIFF"][k] = 0.0
        else:
            p["diff_z"] = 0.0
            p["Z_DIFF"] = {k: 0.0 for k in z_scores_d.keys()}

        # 4. MPG
        # RAW_DARKO has 'mpg' (from darko_services)
        raw_d = p.get("RAW_DARKO", {})
        p["mpg"] = raw_d.get("mpg", 0.0)

        ranked_players.append(p)

    # Filter to top 250 by Real Z-score (for players with real data)
    players_with_real = [p for p in ranked_players if p.get("has_real_data", False)]
    players_without_real = [p for p in ranked_players if not p.get("has_real_data", False)]
    
    # Sort players with real data by real z-score
    players_with_real.sort(key=lambda x: x["total_z_real"], reverse=True)
    
    # Take top 250 with real data
    top_250_real = players_with_real[:250]
    
    # Final list: top 250 by real z-score, sorted by DARKO Z
    final_players = top_250_real
    final_players.sort(key=lambda x: x["total_z_darko"], reverse=True)
    
    # Add rank index
    for i, p in enumerate(final_players):
        p["rank"] = i + 1

    return render_template('rankings.html', players=final_players)
