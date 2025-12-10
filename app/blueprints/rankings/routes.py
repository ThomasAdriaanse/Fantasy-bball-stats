
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
        # If no real data, difference might be misleading. Let's set it to 0 or None.
        if has_real_data:
            p["diff_z"] = total_d - total_r
        else:
            p["diff_z"] = 0.0

        ranked_players.append(p)

    # Sort descending by Total DARKO Z
    ranked_players.sort(key=lambda x: x["total_z_darko"], reverse=True)
    
    # Add rank index
    for i, p in enumerate(ranked_players):
        p["rank"] = i + 1

    return render_template('rankings.html', players=ranked_players)
