from flask import Flask, render_template, redirect, url_for, request, flash, session, send_from_directory, jsonify
import os
from dotenv import load_dotenv
from psycopg2 import extras
import compare_page.compare_page_data as cpd
import compare_page.team_stats_data as tsd
import compare_page.team_cat_averages as tca
from scipy.stats import norm
from espn_api.basketball import League
from espn_api.requests.espn_requests import ESPNUnknownError, ESPNAccessDenied, ESPNInvalidLeague
import pandas as pd
import db_utils
import boto3
from botocore.exceptions import ClientError
# nba_api.stats.endpoints import playergamelog, playercareerstats, leaguegamefinder
from nba_api.stats.static import players

# S3 config (env or defaults)
S3_BUCKET = os.getenv("S3_BUCKET", "fantasy-stats-dev")
S3_PREFIX = os.getenv("S3_PREFIX", "dev/players/")  # includes trailing slash
AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")

import json
import time
from pprint import pprint
from datetime import datetime, timedelta, date

load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-not-secret-change-me")
app.permanent_session_lifetime = timedelta(days=1)
# ---------- Utilities ----------

def _parse_league_details_from_request(req):
    """Best-effort parse of league details from GET or POST."""
    lid  = (req.values.get('league_id') or "").strip()
    yr   = (req.values.get('year') or "").strip()
    s2   = (req.values.get('espn_s2') or "").strip() or None
    swid = (req.values.get('swid') or "").strip() or None

    if not lid or not yr:
        return None  # insufficient to override

    try:
        yr = int(yr)
    except ValueError:
        return None

    return {'league_id': lid, 'year': yr, 'espn_s2': s2, 'swid': swid}


def _store_league_details(details: dict):
    """Write normalized league details into the session; always mark as modified."""
    if not details:
        return
    # Normalize
    league_id = (details.get('league_id') or '').strip()
    year = details.get('year')
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None

    new_payload = {
        'league_id': league_id or None,
        'year': year,
        'espn_s2': (details.get('espn_s2') or None) or None,
        'swid': (details.get('swid') or None) or None,
    }

    session.permanent = True
    session['league_details'] = new_payload
    # Optional: keep a simple change token to detect flips (useful for debugging)
    session['league_changed_at'] = datetime.utcnow().isoformat()
    session.modified = True

@app.before_request
def _capture_league_from_request():
    # Hard reset: /?reset=1
    if request.args.get('reset') in ('1', 'true', 'True'):
        session.pop('league_details', None)
        session.modified = True
        return  # don’t attempt to parse after reset

    # Legacy ?info=league,year,espn_s2,swid → ALWAYS override if present
    if 'info' in request.args:
        parts = request.args.get('info', '').split(',')
        if len(parts) >= 2:
            try:
                year = int(parts[1])
            except Exception:
                year = None
            legacy = {
                'league_id': parts[0] if parts and parts[0] else None,
                'year': year,
                'espn_s2': parts[2] or None if len(parts) > 2 else None,
                'swid'   : parts[3] or None if len(parts) > 3 else None,
            }
            _store_league_details(legacy)

    # Explicit form/query params take precedence if both league_id and year exist
    explicit = _parse_league_details_from_request(request)
    if explicit and explicit.get('league_id') and explicit.get('year'):
        _store_league_details(explicit)


def get_matchup_dates(league, league_year):
    """Generate matchup date data for all matchup periods in the league"""
    
    today = date.today()
    current_year = today.year

    if league_year < 2026:
        #if today > datetime(current_year, 3, 30) and today < datetime(current_year, 10, 20):
        season_start = date(2025, 4, 13) - timedelta(days=league.scoringPeriodId)
    else:
        if today < date(2025, 10, 20):
            season_start = date(2025, 10, 20)        
        else:
            season_start = today - timedelta(days=league.scoringPeriodId)
    
    matchupperiods = league.settings.matchup_periods
    first_scoring_period = league.firstScoringPeriod

    matchup_date_data = {}
    matchup_period_keys = [int(k) for k in matchupperiods.keys()]
    max_matchup_period = max(matchup_period_keys)
    
    prev_end_date = None
    scoring_period_multiplier = 0

    for matchup_period_number, matchup_periods in matchupperiods.items():
        matchup_period_number = int(matchup_period_number)
        matchup_length = len(matchup_periods)

        if matchup_period_number == (18 - first_scoring_period):
            matchup_length += 1

        if matchup_period_number == 1:
            start_date = season_start
            end_date = start_date + timedelta(days=6) + timedelta(days=7)*(matchup_length-1)  # First week is shorter
        else:
            start_date = prev_end_date + timedelta(days=1)
            end_date = start_date + timedelta(days=6) + timedelta(days=7)*(matchup_length-1)

        scoring_periods = [i + (matchup_period_number - 1 + scoring_period_multiplier) * 7
                           for i in range((end_date - start_date).days + 1)]
        
        matchup_date_data[f'matchup_{matchup_period_number}'] = {
            'matchup_period': matchup_period_number,
            'scoring_periods': scoring_periods,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }

        scoring_period_multiplier += matchup_length - 1
        prev_end_date = end_date

    return matchup_date_data

# ---------- Routes ----------

@app.route('/')
def entry_page():
    error_message = request.args.get('error_message', '')
    return render_template('entry.html', error_message=error_message)

@app.route('/player_stats', methods=['GET', 'POST'])
def player_stats():
    
    
    # --- Prefer explicit request values over existing session ---
    new_details = _parse_league_details_from_request(request)
    if new_details:
        _store_league_details(new_details)

    num_games = 20
    selected_player = 'Bol Bol'
    selected_stat = 'FPTS'

    if request.method == 'GET':
        selected_player = request.args.get('player_name', selected_player)
        num_games = int(request.args.get('num_games', num_games))
        selected_stat = request.args.get('stat', selected_stat)
    else:
        selected_player = request.form.get('player_name', selected_player)
        num_games = int(request.form.get('num_games', num_games))
        selected_stat = request.form.get('stat', selected_stat)

    stat_options = [
        'FPTS','PTS','REB','AST','STL','BLK','TOV',
        'FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT',
        'FTM','FTA','FT_PCT','MIN','PLUS_MINUS',
        'AVG_Z','Z_PTS','Z_FG3M','Z_REB','Z_AST','Z_STL','Z_BLK','Z_FG_PCT','Z_FT_PCT','Z_TOV'
    ]

    # Build chart data in memory (no file)
    chart_data = build_chart_data(selected_player, num_games, stat=selected_stat)

    active_players = players.get_active_players()
    active_players.sort(key=lambda x: x['full_name'])

    return render_template(
        'player_stats.html',
        players=active_players,
        selected_player=selected_player,
        num_games=num_games,
        stat_options=stat_options,
        selected_stat=selected_stat,
        chart_data_json=json.dumps(chart_data)  # pass to template for inline use
    )

@app.get('/api/player_stats')
def api_player_stats():
    player_name = request.args.get('player_name', 'Bol Bol')
    num_games   = int(request.args.get('num_games', 20))
    stat        = request.args.get('stat', 'FPTS')
    data = build_chart_data(player_name, num_games, stat)
    return jsonify(data)

# Optional compat: serve the same data at /data/player_stats.json if your front-end expects that path.
@app.get('/data/player_stats.json')
def data_player_stats_json():
    player_name = request.args.get('player_name', 'Bol Bol')
    num_games   = int(request.args.get('num_games', 20))
    stat        = request.args.get('stat', 'FPTS')
    data = build_chart_data(player_name, num_games, stat)
    return jsonify(data)

def _safe_filename(name: str) -> str:
    """lowercase-and-underscore a name to match your exported filenames"""
    return ''.join(ch if ch.isalnum() else '_' for ch in name).strip('_').lower()

def _load_player_dataset_from_s3(player_name: str) -> pd.DataFrame | None:
    """
    Loads the per-player JSON you exported earlier:
      s3://<bucket>/<prefix>/<slug>.json
    Returns a DataFrame with the 'rows' content from that JSON.
    """
    slug = _safe_filename(player_name)
    key  = f"{S3_PREFIX}{slug}.json"

    s3 = boto3.client("s3", region_name=AWS_REGION)

    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "Unknown")
        print(f"[S3] get_object failed for s3://{S3_BUCKET}/{key} ({code})")
        return None

    try:
        payload = json.loads(obj["Body"].read())
    except Exception as e:
        print(f"[S3] JSON decode error for {key}: {e}")
        return None

    rows = payload.get("rows", [])
    if not rows:
        print(f"[S3] No 'rows' in payload for {key}")
        return None

    df = pd.DataFrame(rows)

    # Light numeric coercion (don’t clobber strings like MATCHUP/SEASON)
    numeric_candidates = [c for c in df.columns if c not in ("MATCHUP","SEASON","SEASON_ID","GAME_DATE","GAME_DATE_TS")]
    for c in numeric_candidates:
        df[c] = pd.to_numeric(df[c])

    # Ensure ordering columns exist
    if "Game_Number" not in df.columns and "GAME_DATE_TS" in df.columns:
        df = df.sort_values("GAME_DATE_TS").reset_index(drop=True)
        df["Game_Number"] = df.index + 1

    return df

# ---------- Core: single-player chart JSON ----------
def build_chart_data(player_name: str, num_games: int, stat: str = "FPTS"):
    """
    Load player's full dataset from S3 and return compact chart data
    for the selected stat as a Python list (no files written).
    """
    df = _load_player_dataset_from_s3(player_name)
    if df is None or df.empty:
        print(f"[S3] No data for {player_name}")
        return []

    required_meta = ["Game_Number", "MATCHUP", "SEASON", "SEASON_ID"]
    for col in required_meta:
        if col not in df.columns:
            df[col] = None

    base_allowed = {
        'FPTS','PTS','REB','AST','STL','BLK','TOV',
        'FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT',
        'FTM','FTA','FT_PCT','MIN','PLUS_MINUS'
    }
    z_allowed = {'AVG_Z'} | {f'Z_{k}' for k in [
        'PTS','FG3M','REB','AST','STL','BLK','FG_PCT','FT_PCT','TOV'
    ]}
    value_col = stat if stat in (base_allowed | z_allowed) else 'FPTS'
    if value_col not in df.columns:
        print(f"[WARN] '{value_col}' missing for {player_name}; falling back to FPTS.")
        value_col = "FPTS"
        if value_col not in df.columns:
            return []

    df = df.sort_values("Game_Number").reset_index(drop=True)

    def centered_average(idx, half_window):
        start = max(0, idx - half_window)
        end   = min(len(df), idx + half_window + 1)
        return pd.to_numeric(df[value_col].iloc[start:end], errors="coerce").mean()

    df['Centered_Avg'] = df.index.map(lambda i: centered_average(i, num_games))
    chart_df = df[['Game_Number', 'MATCHUP', 'SEASON', 'SEASON_ID']].copy()
    chart_df['Centered_Avg_Stat'] = pd.to_numeric(df['Centered_Avg'], errors='coerce').round(3)
    chart_df['STAT'] = value_col
    chart_df = chart_df.dropna(subset=['Centered_Avg_Stat'])

    return chart_df[['Game_Number','Centered_Avg_Stat','MATCHUP','SEASON','SEASON_ID','STAT']].to_dict(orient='records')

# ---------- Compare page (unchanged) ----------

def calculate_fantasy_points(row, scoring_rules):
    fpts = (row['FGM'] * scoring_rules['fgm'] +
            row['FGA'] * scoring_rules['fga'] +
            row['FTM'] * scoring_rules['ftm'] +
            row['FTA'] * scoring_rules['fta'] +
            row['FG3M'] * scoring_rules['threeptm'] +
            row['REB'] * scoring_rules['reb'] +
            row['AST'] * scoring_rules['ast'] +
            row['STL'] * scoring_rules['stl'] +
            row['BLK'] * scoring_rules['blk'] +
            row['TOV'] * scoring_rules['turno'] +
            row['PTS'] * scoring_rules['pts'])
    return fpts
@app.route('/compare_page', methods=['POST'])
def compare_page():
    start_time = time.time()
    try:
        # Attempt to convert form inputs to integers
        fgm = int(request.form.get('fgm', 2))
        fga = int(request.form.get('fga', -1))
        ftm = int(request.form.get('ftm', 1))
        fta = int(request.form.get('fta', -1))
        threeptm = int(request.form.get('threeptm', 1))
        reb = int(request.form.get('reb', 1))
        ast = int(request.form.get('ast', 2))
        stl = int(request.form.get('stl', 4))
        blk = int(request.form.get('blk', 4))
        turno = int(request.form.get('turno', -2))
        pts = int(request.form.get('pts', 1))
    except (ValueError, TypeError):
        flash("Invalid input. Please ensure all stats are numbers.")
        league_id = request.form.get('league_id')
        year = request.form.get('year')
        espn_s2 = request.form.get('espn_s2')
        swid = request.form.get('swid')
        scoring_type = request.form.get('scoring_type')
        info_list = [league_id, year, espn_s2, swid]
        info_string = ','.join(filter(None, info_list))
        session['form_data'] = request.form.to_dict()
        return redirect(url_for('select_teams_page', info=info_string, scoring_type=scoring_type))

    # One stat window for BOTH teams (from the form)
    raw_window = (request.form.get('stat_window') or 'projected').strip().lower().replace('-', '_')
    VALID_WINDOWS = {'projected', 'total', 'last_30', 'last_15', 'last_7'}
    stat_window = raw_window if raw_window in VALID_WINDOWS else 'projected'

    league_scoring_rules = {
        'fgm': fgm, 'fga': fga, 'ftm': ftm, 'fta': fta,
        'threeptm': threeptm, 'reb': reb, 'ast': ast, 'stl': stl,
        'blk': blk, 'turno': turno, 'pts': pts,
    }

    my_team_name = request.form.get('myTeam')
    opponents_team_name = request.form.get('opponentsTeam')
    league_id = request.form.get('league_id')
    year = int(request.form.get('year'))
    espn_s2 = request.form.get('espn_s2')
    swid = request.form.get('swid')
    scoring_type = request.form.get('scoring_type')
    week_num = int(request.form.get('week_num'))

    _store_league_details({'league_id': league_id, 'year': year, 'espn_s2': espn_s2, 'swid': swid})

    print(f"Processing league with scoring type: {scoring_type}, stat_window={stat_window}")

    try:
        if espn_s2 and swid:
            league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
        else:
            league = League(league_id=league_id, year=year)

        matchup_data_dict = get_matchup_dates(league, year)
        selected_week_key = f'matchup_{week_num}'
        week_data = {
            "selected_week": week_num,
            "current_week": league.currentMatchupPeriod,
            "matchup_data": matchup_data_dict.get(selected_week_key, {})
        }
    except (ESPNUnknownError, ESPNInvalidLeague, ESPNAccessDenied) as e:
        error_message = "Error accessing ESPN league. Please check your league ID and credentials."
        if isinstance(e, ESPNAccessDenied):
            error_message = "This is a private league. Please provide ESPN S2 and SWID credentials."
        return redirect(url_for('entry_page', error_message=error_message))
    except Exception as e:
        print(f"Error initializing league: {str(e)}")
        return redirect(url_for('entry_page', error_message=str(e)))

    # Find team indices
    team1_index = -1
    team2_index = -1
    for idx, t in enumerate(league.teams):
        if t.team_name == my_team_name:
            team1_index = idx
        if t.team_name == opponents_team_name:
            team2_index = idx

    if team1_index == -1:
        return redirect(url_for('entry_page', error_message="Team 1 not found."))
    if team2_index == -1:
        return redirect(url_for('entry_page', error_message="Team 2 not found."))

    player_data_column_names = [
        'player_name','min','fgm','fga','fg%','ftm','fta','ft%','threeptm',
        'reb','ast','stl','blk','turno','pts','inj','fpts','games'
    ]

    if scoring_type == "H2H_POINTS":
        # Use the SAME stat_window for both teams
        team1_player_data = cpd.get_team_player_data(
            league, team1_index, player_data_column_names, year,
            league_scoring_rules, week_data, stat_window=stat_window
        )
        team2_player_data = cpd.get_team_player_data(
            league, team2_index, player_data_column_names, year,
            league_scoring_rules, week_data, stat_window=stat_window
        )

        team_data_column_names = [
            'team_avg_fpts', 'team_expected_points', 'team_chance_of_winning',
            'team_name', 'team_current_points'
        ]
        team1_data, team2_data = tsd.get_team_stats(
            league, team1_index, team1_player_data,
            team2_index, team2_player_data,
            team_data_column_names, league_scoring_rules, year, week_data
        )

        combined_df = cpd.get_compare_graph(
            league, team1_index, team1_player_data,
            team2_index, team2_player_data, year, week_data
        )
        combined_json = combined_df.to_json(orient='records')

        return render_template(
            'compare_page.html',
            data_team_players_1=team1_player_data.to_dict(orient='records'),
            data_team_players_2=team2_player_data.to_dict(orient='records'),
            data_team_stats_1=team1_data.to_dict(orient='records'),
            data_team_stats_2=team2_data.to_dict(orient='records'),
            combined_json=combined_json,
            scoring_type="H2H_POINTS",
            week_data=week_data,
            stat_window=stat_window,
        )

    elif scoring_type in ["H2H_CATEGORY", "H2H_MOST_CATEGORIES"]:
        # Use the SAME stat_window for both teams
        team1_player_data = cpd.get_team_player_data(
            league, team1_index, player_data_column_names, year,
            league_scoring_rules, week_data, stat_window=stat_window
        )
        team2_player_data = cpd.get_team_player_data(
            league, team2_index, player_data_column_names, year,
            league_scoring_rules, week_data, stat_window=stat_window
        )

        (team1_data, team2_data,
         team1_win_pcts, team2_win_pcts,
         team1_current_stats, team2_current_stats) = tsd.get_team_stats_categories(
            league, team1_index, team1_player_data,
            team2_index, team2_player_data,
            league_scoring_rules, year, week_data
        )

        combined_dfs = cpd.get_compare_graphs_categories(
            league, team1_index, team1_player_data,
            team2_index, team2_player_data, year, week_data
        )
        combined_dicts = {cat: df.to_dict(orient='records') for cat, df in combined_dfs.items()}

        return render_template(
            'compare_page_cat.html',
            data_team_players_1=team1_player_data.to_dict(orient='records'),
            data_team_players_2=team2_player_data.to_dict(orient='records'),
            data_team_stats_1=team1_data.to_dict(orient='records'),
            data_team_stats_2=team2_data.to_dict(orient='records'),
            team1_win_pct_data=team1_win_pcts.to_dict(orient='records'),
            team2_win_pct_data=team2_win_pcts.to_dict(orient='records'),
            team1_current_stats=team1_current_stats,
            team2_current_stats=team2_current_stats,
            combined_jsons=combined_dicts,
            scoring_type=scoring_type,
            week_data=week_data,
            stat_window=stat_window,
        )

    print(f"Unrecognized scoring type: {scoring_type}")
    return redirect(url_for('entry_page', error_message=f"Unsupported scoring type: {scoring_type}"))


@app.route('/select_teams_page')
def select_teams_page():
    # by now, before_request has already applied ?info= or explicit params if present
    league_details = session.get('league_details') or {}
    if not league_details.get('league_id') or not league_details.get('year'):
        return redirect(url_for('entry_page', error_message="Enter your league first."))

    try:
        league = League(
            league_id=league_details['league_id'],
            year=league_details['year'],
            espn_s2=league_details.get('espn_s2'),
            swid=league_details.get('swid')
        ) if (league_details.get('espn_s2') and league_details.get('swid')) else League(
            league_id=league_details['league_id'],
            year=league_details['year']
        )
    except (ESPNUnknownError, ESPNInvalidLeague):
        return redirect(url_for('entry_page', error_message="Invalid league entered."))
    except ESPNAccessDenied:
        return redirect(url_for('entry_page', error_message="That is a private league which needs ESPN S2 and SWID."))
    except Exception as e:
        return redirect(url_for('entry_page', error_message=str(e)))

    year = league_details['year']
    matchup_date_data = get_matchup_dates(league, year)
    teams_list = [team.team_name for team in league.teams]
    form_data = session.pop('form_data', {})

    return render_template(
        'select_teams_page.html',
        info_list=teams_list,
        league_id=league_details['league_id'],
        year=league_details['year'],
        espn_s2=league_details.get('espn_s2'),
        swid=league_details.get('swid'),
        form_data=form_data,
        scoring_type=league.settings.scoring_type,
        matchup_data_dict=matchup_date_data,
        current_matchup=league.currentMatchupPeriod
    )




@app.route('/process', methods=['POST'])
def process_information():
    details = _parse_league_details_from_request(request)
    if not details or not details.get('league_id') or not details.get('year'):
        return redirect(url_for('entry_page', error_message="Invalid league entered. Please try again."))
    _store_league_details(details)
    return redirect(url_for('select_teams_page'))


@app.get('/punting_overview')
def punting_overview():
    # accept overrides via query like the rest of your app
    new_details = _parse_league_details_from_request(request)
    if new_details:
        _store_league_details(new_details)

    league_details = session.get('league_details') or {}
    league_id = league_details.get('league_id')
    year      = league_details.get('year')
    espn_s2   = league_details.get('espn_s2')
    swid      = league_details.get('swid')

    if not league_id or not year:
        return redirect(url_for('entry_page', error_message="Enter your league first."))

    stat_window = (request.args.get('stat_window') or 'projected').strip().lower().replace('-', '_')

    try:
        league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid) if espn_s2 and swid \
                 else League(league_id=league_id, year=year)
    except (ESPNUnknownError, ESPNInvalidLeague, ESPNAccessDenied) as e:
        return redirect(url_for('entry_page', error_message=str(e)))

    data = tca._team_category_averages(league, year, stat_window=stat_window)
    # Render template with embedded JSON
    return render_template('punting_overview.html',
                           league_id=league_id, year=year,
                           stat_window=stat_window,
                           data_json=json.dumps(data))

@app.get('/api/punting_overview')
def api_punting_overview():
    league_details = session.get('league_details') or {}
    league_id = league_details.get('league_id')
    year      = league_details.get('year')
    espn_s2   = league_details.get('espn_s2')
    swid      = league_details.get('swid')

    if not league_id or not year:
        return jsonify({'error': 'No league in session'}), 400

    stat_window = (request.args.get('stat_window') or 'projected').strip().lower().replace('-', '_')

    try:
        league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid) if espn_s2 and swid \
                 else League(league_id=league_id, year=year)
    except (ESPNUnknownError, ESPNInvalidLeague, ESPNAccessDenied) as e:
        return jsonify({'error': str(e)}), 400

    data = tca._team_category_averages(league, year, stat_window=stat_window)
    return jsonify(data)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
