from flask import Blueprint, render_template, redirect, url_for, request
from ...services.league_session import _parse_from_req, _store

bp = Blueprint("main", __name__)

@bp.get("/")
def entry_page():
    error_message = request.args.get('error_message', '')
    return render_template("entry.html", error_message=error_message)

@bp.post("/process")
def process_information():
    details = _parse_from_req(request)
    if not details or not details.get('league_id') or not details.get('year'):
        return redirect(url_for('main.entry_page', error_message="Invalid league entered. Please try again."))
    _store(details)
    return redirect(url_for('compare.select_teams_page'))
