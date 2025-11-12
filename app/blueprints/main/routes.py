from flask import Blueprint, render_template, redirect, url_for, request, session

bp = Blueprint("main", __name__)

@bp.get("/")
def entry_page():
    error_message = request.args.get('error_message', '')
    return render_template("entry.html", error_message=error_message)

@bp.post("/process")
def process_information():
    
    league_id = request.form.get("league_id")
    year = request.form.get("year")
    espn_s2 = request.form.get("espn_s2") or None
    swid = request.form.get("swid") or None

    session['league_details'] = {'league_id': league_id, 'year': int(year), 'espn_s2': espn_s2, 'swid': swid}

    return redirect(url_for('compare.select_teams_page'))
