from flask import request, session
from datetime import datetime

def _parse_from_req(req):
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

def _store(details: dict):
    if not details: return
    payload = {
        'league_id': (details.get('league_id') or '').strip() or None,
        'year': int(details.get('year')) if details.get('year') else None,
        'espn_s2': details.get('espn_s2') or None,
        'swid': details.get('swid') or None,
    }
    session.permanent = True
    session['league_details'] = payload
    session['league_changed_at'] = datetime.utcnow().isoformat()
    session.modified = True

def register_league_hooks(app):
    @app.before_request
    def _capture():
        if request.args.get('reset') in ('1','true','True'):
            session.pop('league_details', None); session.modified = True; return

        if 'info' in request.args:
            parts = request.args.get('info','').split(',')
            if len(parts) >= 2:
                try: year = int(parts[1])
                except: year = None
                _store({
                    'league_id': parts[0] or None,
                    'year': year,
                    'espn_s2': parts[2] or None if len(parts) > 2 else None,
                    'swid'   : parts[3] or None if len(parts) > 3 else None,
                })

        explicit = _parse_from_req(request)
        if explicit and explicit.get('league_id') and explicit.get('year'):
            _store(explicit)
