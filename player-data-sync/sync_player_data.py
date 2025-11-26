#!/usr/bin/env python3
"""
Player Data Sync Service
Syncs player JSON data from S3 to local cache directory.

Environment variables:
    S3_BUCKET: S3 bucket name (required)
    S3_PREFIX: S3 prefix for player data (default: dev/players/)
    PLAYER_DATA_CACHE_DIR: Local cache directory (default: /app/data/players)
    AWS_REGION: AWS region (default: ca-central-1)
    AWS_ACCESS_KEY_ID: AWS access key (required)
    AWS_SECRET_ACCESS_KEY: AWS secret key (required)
"""

import os
import sys
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from datetime import datetime

# Configuration
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "dev/players/")
LOCAL_CACHE_DIR = os.getenv("PLAYER_DATA_CACHE_DIR", "/app/data/players")
AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")

# Validate required environment variables
if not S3_BUCKET:
    print("[ERROR] S3_BUCKET environment variable is required")
    sys.exit(1)


def sync_s3_to_local():
    """
    Sync all player JSON files from S3 to local cache directory.
    """
    print(f"[SYNC] Starting sync at {datetime.now().isoformat()}")
    print(f"[SYNC] Source: s3://{S3_BUCKET}/{S3_PREFIX}")
    print(f"[SYNC] Destination: {LOCAL_CACHE_DIR}")
    
    # Create local cache directory if it doesn't exist
    cache_path = Path(LOCAL_CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"[SYNC] Cache directory ready: {cache_path}")
    
    # Initialize S3 client
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        print(f"[SYNC] Connected to S3 in region: {AWS_REGION}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize S3 client: {e}")
        return False
    
    # List all objects in the S3 prefix
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
                
                # Skip if it's just the prefix (directory)
                if key == S3_PREFIX or key.endswith('/'):
                    continue
                
                # Extract filename from key
                filename = key[len(S3_PREFIX):]
                
                # Skip non-JSON files
                if not filename.endswith('.json'):
                    continue
                
                total_files += 1
                local_file_path = cache_path / filename
                
                # Check if file already exists and is up-to-date
                if local_file_path.exists():
                    local_mtime = local_file_path.stat().st_mtime
                    s3_mtime = obj['LastModified'].timestamp()
                    
                    if local_mtime >= s3_mtime:
                        skipped_files += 1
                        if total_files % 100 == 0:
                            print(f"[SYNC] Progress: {total_files} files ({downloaded_files} new, {skipped_files} skipped)")
                        continue
                
                # Download file from S3
                try:
                    s3.download_file(S3_BUCKET, key, str(local_file_path))
                    downloaded_files += 1
                    
                    if downloaded_files % 50 == 0:
                        print(f"[SYNC] Downloaded {downloaded_files} files...")
                        
                except ClientError as e:
                    print(f"[ERROR] Failed to download {key}: {e}")
                    failed_files += 1
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"[SYNC] ✓ Sync completed at {datetime.now().isoformat()}")
        print(f"{'='*60}")
        print(f"  Total files found:     {total_files}")
        print(f"  Downloaded (new):      {downloaded_files}")
        print(f"  Skipped (up-to-date):  {skipped_files}")
        print(f"  Failed:                {failed_files}")
        print(f"{'='*60}")
        
        if failed_files > 0:
            print(f"[WARN] {failed_files} files failed to download")
            return False
        
        if total_files == 0:
            print(f"[WARN] No JSON files found in s3://{S3_BUCKET}/{S3_PREFIX}")
            return False
        
        return True
        
    except ClientError as e:
        print(f"[ERROR] Failed to list S3 objects: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error during sync: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("Player Data Sync Service")
    print("="*60)
    
    success = sync_s3_to_local()
    
    if success:
        print("\n[SUCCESS] Sync completed successfully")
        sys.exit(0)
    else:
        print("\n[FAILURE] Sync failed")
        sys.exit(1)
