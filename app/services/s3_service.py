# app/services/s3_service.py
import os
import json
import pandas as pd
import boto3
from botocore.exceptions import ClientError

# S3 config (env or defaults)
S3_BUCKET = os.getenv("S3_BUCKET", "fantasy-stats-dev")
S3_PREFIX = os.getenv("S3_PREFIX", "dev/players/")  # includes trailing slash
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "ca-central-1"))

# Local cache directory (can be set via environment variable)
LOCAL_CACHE_DIR = os.getenv("PLAYER_DATA_CACHE_DIR", "/app/data/players")

# Simple in-memory cache to avoid reloading same player multiple times
_PLAYER_CACHE = {}

def _safe_filename(name: str) -> str:
    """
    Convert player name to filename format.
    Replace spaces, hyphens, periods, apostrophes with underscores.
    Preserve Unicode letters like č, ć, ñ, é, etc.
    """
    # Replace common punctuation/separators with underscore
    result = name.replace(" ", "_").replace("-", "_").replace(".", "_").replace("'", "_")
    # Remove any other non-alphanumeric except underscores (but keep Unicode letters)
    result = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in result)
    # Strip leading/trailing underscores only
    return result.strip("_").lower()

def load_player_dataset_from_s3(player_name: str) -> pd.DataFrame | None:
    """
    Loads the per-player JSON from local cache or S3:
      Local: {LOCAL_CACHE_DIR}/<slug>.json
      S3: s3://<bucket>/<prefix>/<slug>.json
    Returns a DataFrame with the 'rows' content from that JSON.
    Uses in-memory caching to avoid reloading same player multiple times.
    """
    # Check in-memory cache first
    if player_name in _PLAYER_CACHE:
        return _PLAYER_CACHE[player_name]
    
    slug = _safe_filename(player_name)
    filename = f"{slug}.json"
    
    # Try local cache first
    local_path = os.path.join(LOCAL_CACHE_DIR, filename)
    if os.path.exists(local_path):
        try:
            with open(local_path, 'r') as f:
                payload = json.load(f)
            df = _process_player_data(payload, player_name, filename)
            _PLAYER_CACHE[player_name] = df
            return df
        except Exception as e:
            print(f"[LOCAL] Error reading {local_path}: {e}, falling back to S3")
    
    # Fall back to S3 if not in local cache
    key = f"{S3_PREFIX}{filename}"

    s3 = boto3.client("s3", region_name=AWS_REGION)

    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "Unknown")
        print(f"[S3] get_object failed for s3://{S3_BUCKET}/{key} ({code})")
        # Cache the failure too (as None) to avoid retrying
        _PLAYER_CACHE[player_name] = None
        return None

    try:
        payload = json.loads(obj["Body"].read())
    except Exception as e:
        print(f"[S3] JSON decode error for {key}: {e}")
        _PLAYER_CACHE[player_name] = None
        return None

    df = _process_player_data(payload, player_name, key)
    _PLAYER_CACHE[player_name] = df
    return df


def _process_player_data(payload: dict, player_name: str, source: str) -> pd.DataFrame | None:
    """
    Process player JSON payload into a DataFrame.
    
    Args:
        payload: JSON payload with 'rows' key
        player_name: Player name for caching
        source: Source identifier (for logging)
    
    Returns:
        Processed DataFrame or None if invalid
    """
    rows = payload.get("rows", [])
    if not rows:
        print(f"[DATA] No 'rows' in payload for {source}")
        return None

    df = pd.DataFrame(rows)

    # Light numeric coercion (don't clobber known string/date cols)
    non_numeric_cols = {"MATCHUP", "SEASON", "SEASON_ID", "GAME_DATE", "GAME_DATE_TS"}
    numeric_candidates = [c for c in df.columns if c not in non_numeric_cols]
    for c in numeric_candidates:
        # Use 'coerce' instead of deprecated errors="ignore" to avoid FutureWarning
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure ordering columns exist
    if "Game_Number" not in df.columns:
        if "GAME_DATE_TS" in df.columns:
            df = df.sort_values("GAME_DATE_TS").reset_index(drop=True)
        else:
            # fall back to original order
            df = df.reset_index(drop=True)
        df["Game_Number"] = df.index + 1

    return df
