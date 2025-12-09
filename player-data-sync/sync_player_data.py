#!/usr/bin/env python3
"""
Player Data Sync Service

Modes (via SYNC_MODE env var):
    SYNC_MODE=sync  -> Only sync player JSONs from S3 to local cache (old behavior).
    SYNC_MODE=pmf   -> Only build per-player PMF files from existing local JSONs.
    SYNC_MODE=both  -> First sync from S3, then build PMFs.

Environment variables:
    S3_BUCKET: S3 bucket name (required for sync / both)
    S3_PREFIX: S3 prefix for player data (default: dev/players/)
    PLAYER_DATA_CACHE_DIR: Local cache dir for JSON (default: /app/data/players)
    PLAYER_PMF_CACHE_DIR: Local cache dir for PMFs (default: /app/data/pmf)
    AWS_REGION: AWS region (default: ca-central-1)
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY: AWS creds (required for sync / both)
    SYNC_MODE: one of "sync", "pmf", "both" (default: "sync")
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
import numpy as np

# ============================================================
#  INLINE MINIMAL PMF CLASSES (1D AND 2D)
# ============================================================

class PMF:
    """
    Simple 1D PMF backed by numpy.
    probs[i] represents P(X = i)
    """

    def __init__(self, data):
        """
        data can be:
            dict[value] = prob
            or numpy array of counts/probs
        """
        if isinstance(data, dict):
            max_key = max(data.keys()) if data else 0
            arr = np.zeros(max_key + 1, dtype=float)
            for k, v in data.items():
                arr[int(k)] = float(v)
        else:
            arr = np.asarray(data, dtype=float)

        s = arr.sum()
        if s > 0:
            arr = arr / s

        self.arr = arr

    @staticmethod
    def _from_array(arr):
        return PMF(arr)

    def _to_array(self):
        return self.arr


class PMF2D:
    """
    Simple 2D PMF backed by numpy.
    probs[i,j] = P(X=i, Y=j)
    """

    def __init__(self, data):
        arr = np.asarray(data, dtype=float)
        s = arr.sum()
        if s > 0:
            arr = arr / s
        self.arr = arr

    @staticmethod
    def _from_array(arr):
        return PMF2D(arr)

    def _to_array(self):
        return self.arr


# =========================
# Configuration
# =========================
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "dev/players/")
LOCAL_CACHE_DIR = os.getenv("PLAYER_DATA_CACHE_DIR", "/app/data/players")
PMF_CACHE_DIR = os.getenv("PLAYER_PMF_CACHE_DIR", "/app/data/pmf")
AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")
SYNC_MODE = os.getenv("SYNC_MODE", "sync").lower()

# 1D stats we build PMFs for (single-game distributions)
PMF_1D_STATS = [
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "FG3M",
    "TOV",
]

# 2D stats for FG% and FT% PMFs (makes, attempts)
PMF_2D_STATS = {
    "FG": ("FGM", "FGA"),
    "FT": ("FTM", "FTA"),
}


# ============================================================
#  A) ORIGINAL SYNC FUNCTION (UNCHANGED BEHAVIOR)
# ============================================================

def sync_s3_to_local() -> bool:
    if not S3_BUCKET:
        print("[ERROR] S3_BUCKET environment variable is required for sync")
        return False

    print(f"[SYNC] Starting sync at {datetime.now().isoformat()}")
    print(f"[SYNC] Source: s3://{S3_BUCKET}/{S3_PREFIX}")
    print(f"[SYNC] Destination: {LOCAL_CACHE_DIR}")

    cache_path = Path(LOCAL_CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"[SYNC] Cache directory ready: {cache_path}")

    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        print(f"[SYNC] Connected to S3 in region: {AWS_REGION}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize S3 client: {e}")
        return False

    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX)

        total_files = 0
        downloaded_files = 0
        skipped_files = 0
        failed_files = 0

        print(f"[SYNC] Listing objects in S3...")

        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                key = obj['Key']

                if key == S3_PREFIX or key.endswith('/'):
                    continue

                filename = key[len(S3_PREFIX):]

                if not filename.endswith('.json'):
                    continue

                total_files += 1
                local_file_path = cache_path / filename

                if local_file_path.exists():
                    local_mtime = local_file_path.stat().st_mtime
                    s3_mtime = obj['LastModified'].timestamp()

                    if local_mtime >= s3_mtime:
                        skipped_files += 1
                        if total_files % 100 == 0:
                            print(f"[SYNC] Progress: {total_files} files ({downloaded_files} new, {skipped_files} skipped)")
                        continue

                try:
                    s3.download_file(S3_BUCKET, key, str(local_file_path))
                    downloaded_files += 1

                    if downloaded_files % 50 == 0:
                        print(f"[SYNC] Downloaded {downloaded_files} files...")

                except ClientError as e:
                    print(f"[ERROR] Failed to download {key}: {e}")
                    failed_files += 1

        print(f"\n{'='*60}")
        print(f"[SYNC] ✓ Sync completed at {datetime.now().isoformat()}")
        print(f"{'='*60}")
        print(f"  Total files found:     {total_files}")
        print(f"  Downloaded (new):      {downloaded_files}")
        print(f"  Skipped (up-to-date):  {skipped_files}")
        print(f"  Failed:                {failed_files}")
        print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"[ERROR] Unexpected error during sync: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
#  B) PMF BUILDING
# ============================================================

def _compute_player_pmfs_from_json(json_path: Path, pmf_dir: Path) -> None:
    try:
        with json_path.open("r") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"[PMF] Skipping {json_path.name}: JSON load error: {e}")
        return

    rows = payload.get("rows")
    if not isinstance(rows, list):
        if isinstance(payload, list):
            rows = payload
        else:
            print(f"[PMF] Skipping {json_path.name}: no 'rows' list found")
            return

    # ======== NEW: filter to 2025-26 season only ========
    TARGET_SEASON = "2025-26"
    try:
        start_year = int(TARGET_SEASON.split("-")[0])
        allowed_prefixes = (str(start_year), str(start_year + 1))  # "2025", "2026"
    except Exception:
        # Fallback if the format ever changes
        allowed_prefixes = ("2025", "2026")

    filtered_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        season = str(row.get("SEASON", "")).strip()
        # Only keep rows that have a SEASON and match the 2025-26 window
        if not season:
            continue
        if not season.startswith(allowed_prefixes):
            continue

        filtered_rows.append(row)

    if not filtered_rows:
        print(f"[PMF] Skipping {json_path.name}: no 2025-26 rows found")
        return

    rows = filtered_rows
    # ======== END NEW FILTERING ========

    stat_values = {stat: [] for stat in PMF_1D_STATS}
    fg_makes, fg_attempts = [], []
    ft_makes, ft_attempts = [], []

    for row in rows:
        # 1D stats
        for stat in PMF_1D_STATS:
            v = row.get(stat)
            try:
                stat_values[stat].append(float(v))
            except Exception:
                pass

        # FG
        fgm = row.get("FGM")
        fga = row.get("FGA")
        try:
            fg_makes.append(float(fgm))
            fg_attempts.append(float(fga))
        except Exception:
            pass

        # FT
        ftm = row.get("FTM")
        fta = row.get("FTA")
        try:
            ft_makes.append(float(ftm))
            ft_attempts.append(float(fta))
        except Exception:
            pass

    # ---- 1D PMFs ----
    pmf_1d_out = {}
    for stat, vals in stat_values.items():
        if not vals:
            pmf_obj = PMF({0: 1.0})
        else:
            arr = np.array(vals, dtype=float)
            arr = np.round(arr).astype(int)
            arr = np.clip(arr, 0, None)

            max_val = int(arr.max())
            counts = np.bincount(arr, minlength=max_val + 1).astype(float)
            pmf_obj = PMF._from_array(counts)

        pmf_1d_out[stat] = {"probs": pmf_obj._to_array().tolist()}

    # ---- 2D PMFs ----

    def _build_2d(makes, attempts):
        if not makes or not attempts:
            arr = np.zeros((1, 1), dtype=float)
            arr[0, 0] = 1.0
            return arr

        m_arr = np.array(makes, dtype=float)
        a_arr = np.array(attempts, dtype=float)

        m_arr = np.round(m_arr).astype(int)
        a_arr = np.round(a_arr).astype(int)

        m_arr = np.clip(m_arr, 0, None)
        a_arr = np.clip(a_arr, 0, None)
        m_arr = np.minimum(m_arr, a_arr)

        max_m = int(m_arr.max())
        max_a = int(a_arr.max())
        hist = np.zeros((max_m + 1, max_a + 1), dtype=float)
        for m, a in zip(m_arr, a_arr):
            hist[m, a] += 1.0

        pmf2 = PMF2D._from_array(hist)
        return pmf2._to_array()

    fg_arr = _build_2d(fg_makes, fg_attempts)
    ft_arr = _build_2d(ft_makes, ft_attempts)

    pmf_2d_out = {
        "FG": {"shape": [fg_arr.shape[0], fg_arr.shape[1]], "data": fg_arr.tolist()},
        "FT": {"shape": [ft_arr.shape[0], ft_arr.shape[1]], "data": ft_arr.tolist()},
    }

    pmf_dir.mkdir(parents=True, exist_ok=True)
    pmf_filename = json_path.stem + "_pmf.json"
    pmf_path = pmf_dir / pmf_filename

    obj = {
        "player_file": json_path.name,
        "generated_at": datetime.utcnow().isoformat(),
        "pmf_1d": pmf_1d_out,
        "pmf_2d": pmf_2d_out,
    }

    try:
        with pmf_path.open("w") as f:
            json.dump(obj, f)
    except Exception as e:
        print(f"[PMF] Failed to write PMF file {pmf_path}: {e}")


def build_pmfs_from_local_cache() -> bool:
    cache_path = Path(LOCAL_CACHE_DIR)
    pmf_path = Path(PMF_CACHE_DIR)

    if not cache_path.exists():
        print(f"[PMF] Local cache dir {cache_path} does not exist")
        return False

    pmf_path.mkdir(parents=True, exist_ok=True)
    print(f"[PMF] Building PMFs from local cache:")
    print(f"       JSON dir: {cache_path}")
    print(f"       PMF  dir: {pmf_path}")

    total = 0
    for json_file in cache_path.glob("*.json"):
        total += 1
        _compute_player_pmfs_from_json(json_file, pmf_path)
        if total % 100 == 0:
            print(f"[PMF] Processed {total} player files...")

    print(f"[PMF] Done. PMFs built for {total} players (or attempted).")
    return True


# ============================================================
#  C) MAIN ENTRYPOINT WITH MODE SWITCH
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Player Data Sync Service")
    print("=" * 60)
    print(f"[MODE] SYNC_MODE={SYNC_MODE}")

    if SYNC_MODE == "sync":
        success = sync_s3_to_local()

    elif SYNC_MODE == "pmf":
        success = build_pmfs_from_local_cache()

    elif SYNC_MODE == "both":
        success = sync_s3_to_local()
        if success:
            success = build_pmfs_from_local_cache()

    else:
        print(f"[ERROR] Unknown SYNC_MODE={SYNC_MODE!r}. Use 'sync', 'pmf', or 'both'.")
        sys.exit(1)

    if success:
        print("\n[SUCCESS] Completed successfully")
        sys.exit(0)
    else:
        print("\n[FAILURE] Operation failed")
        sys.exit(1)
