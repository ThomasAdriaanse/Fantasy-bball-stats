# app/blueprints/overview/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, session
import json

from espn_api.basketball import League
from espn_api.requests.espn_requests import ESPNUnknownError, ESPNAccessDenied, ESPNInvalidLeague

# reuse your existing analysis helpers
import compare_page.team_cat_averages as tca

bp = Blueprint("overview", __name__)

# --- small helpers (kept local for simplicity) ---

def _parse_league_details_from_request(req):
    """Best-effort parse of league details from GET/POST."""
    lid  = (req.values.get('league_id') or "").strip()
    yr   = (req.values.get('year') or "").strip()
    s2   = (req.values.get('espn_s2') or "").strip() or None
    swid = (req.values.get('swid') or "").strip() or None

    if not lid or not yr:
        return None
    try:
        yr = int(yr)
    except ValueError:
        return None

    return {'league_id': lid, 'year': yr, 'espn_s2': s2, 'swid': swid}


# =========================
#       Page route
# =========================
@bp.get("/")
def punting_overview():


    league_details = session.get('league_details') or {}
    if not league_details.get('league_id') or not league_details.get('year'):
        return redirect(url_for('main.entry_page', error_message="Enter your league first."))

    league_id = league_details.get('league_id')
    year      = league_details.get('year')

    league = League(
        league_id=league_details['league_id'],
        year=league_details['year'],
        espn_s2=league_details['espn_s2'],
        swid=league_details['swid']
    )

    stat_window = (request.args.get('stat_window') or 'total').strip().lower().replace('-', '_')

    data = tca._team_category_averages(league, year, stat_window=stat_window)

    return render_template(
        'punting_overview.html',
        league_id=league_id,
        year=year,
        stat_window=stat_window,
        data_json=json.dumps(data)
    )


# =========================
#        API route
# =========================
@bp.get("/api")
def punting_overview_api():
    # allow overrides here too
    if err:
        # return JSON error for API
        return jsonify({'error': 'No valid league in session'}), 400

    league_details = session.get('league_details') or {}
    year = league_details.get('year')

    stat_window = (request.args.get('stat_window') or 'projected').strip().lower().replace('-', '_')
    try:
        data = tca._team_category_averages(league, year, stat_window=stat_window)
        return jsonify(data)
    except (ESPNUnknownError, ESPNInvalidLeague, ESPNAccessDenied) as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
