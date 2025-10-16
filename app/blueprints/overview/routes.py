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

def _store_league_details(details: dict):
    if not details:
        return
    league_id = (details.get('league_id') or '').strip() or None
    year = details.get('year')
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None

    payload = {
        'league_id': league_id,
        'year': year,
        'espn_s2': (details.get('espn_s2') or None),
        'swid': (details.get('swid') or None),
    }
    session.permanent = True
    session['league_details'] = payload
    session.modified = True

def _get_league_from_session_or_redirect():
    """Build an ESPN League from session; redirect to entry if missing/invalid."""
    league_details = session.get('league_details') or {}
    league_id = league_details.get('league_id')
    year      = league_details.get('year')
    espn_s2   = league_details.get('espn_s2')
    swid      = league_details.get('swid')

    if not league_id or not year:
        # main.entry_page is the endpoint on your main blueprint
        return None, redirect(url_for('main.entry_page', error_message="Enter your league first."))

    try:
        league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid) if espn_s2 and swid \
                 else League(league_id=league_id, year=year)
        return league, None
    except (ESPNUnknownError, ESPNInvalidLeague, ESPNAccessDenied) as e:
        return None, redirect(url_for('main.entry_page', error_message=str(e)))
    except Exception as e:
        return None, redirect(url_for('main.entry_page', error_message=str(e)))


# =========================
#       Page route
# =========================
@bp.get("/")
def punting_overview():
    # allow ?league_id=&year=&espn_s2=&swid= overrides
    new_details = _parse_league_details_from_request(request)
    if new_details:
        _store_league_details(new_details)

    league, err = _get_league_from_session_or_redirect()
    if err:
        return err

    league_details = session.get('league_details') or {}
    league_id = league_details.get('league_id')
    year      = league_details.get('year')

    stat_window = (request.args.get('stat_window') or 'projected').strip().lower().replace('-', '_')

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
    new_details = _parse_league_details_from_request(request)
    if new_details:
        _store_league_details(new_details)

    league, err = _get_league_from_session_or_redirect()
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
