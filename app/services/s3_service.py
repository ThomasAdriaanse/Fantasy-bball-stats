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

def _safe_filename(name: str) -> str:
    """lowercase-and-underscore a name to match your exported filenames"""
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()

def load_player_dataset_from_s3(player_name: str) -> pd.DataFrame | None:
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

    # Light numeric coercion (don’t clobber known string/date cols)
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
