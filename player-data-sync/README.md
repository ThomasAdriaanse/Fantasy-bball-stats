# Player Data Sync Service

Syncs NBA player data from S3 and generates Probability Mass Functions (PMFs) for fantasy basketball analysis.

## Sync Modes

The sync service has **three modes** controlled by the `SYNC_MODE` environment variable:

| Mode | Description | Use Case |
|------|-------------|----------|
| `sync` | Download player JSON files from S3 | Update raw player data |
| `pmf` | Generate PMF files from existing player data | Regenerate PMFs after code changes |
| `both` | Download from S3 **then** generate PMFs | Full refresh (recommended) |

**Default:** `sync` (for backward compatibility)

## Running the Sync

### Option 1: Download Player Data Only
```bash
docker-compose run --rm sync
# or explicitly:
docker-compose run --rm -e SYNC_MODE=sync sync
```

### Option 2: Generate PMFs Only (from existing player data)
```bash
docker-compose run --rm -e SYNC_MODE=pmf sync
```

### Option 3: Full Refresh (Download + Generate PMFs)
```bash
docker-compose run --rm -e SYNC_MODE=both sync
```

### First-Time Setup
```bash
# 1. Download player data and generate PMFs
docker-compose run --rm -e SYNC_MODE=both sync

# 2. Start the web app
docker-compose up -d web
```

### Complete Refresh (Delete Old Data)
```bash
# 1. Stop the web container
docker-compose down

# 2. Remove old volumes
docker volume rm fantasy-scraper-website_player_data
docker volume rm fantasy-scraper-website_player_data_pmf

# 3. Download and generate everything
docker-compose run --rm -e SYNC_MODE=both sync

# 4. Restart the web app
docker-compose up -d
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SYNC_MODE` | No | `sync` | Mode: `sync`, `pmf`, or `both` |
| `S3_BUCKET` | Yes* | - | S3 bucket name (*required for `sync`/`both`) |
| `S3_PREFIX` | No | `dev/players/` | S3 prefix for player data |
| `PLAYER_DATA_CACHE_DIR` | No | `/app/data/players` | Local cache for player JSON files |
| `PLAYER_PMF_CACHE_DIR` | No | `/app/data/pmf` | Local cache for PMF files |
| `AWS_REGION` | No | `ca-central-1` | AWS region |
| `AWS_ACCESS_KEY_ID` | Yes* | - | AWS access key (*required for `sync`/`both`) |
| `AWS_SECRET_ACCESS_KEY` | Yes* | - | AWS secret key (*required for `sync`/`both`) |

## How It Works

1. **Sync service** downloads all player JSON files from S3 to `/app/data/players` in the Docker volume
2. **Web app** reads from the same volume (read-only) for instant access
3. **Docker volume** persists data between container restarts
4. **Weekly cron** keeps data fresh

## Volume Location

The Docker volume `player_data` is stored on your EC2 instance at:
```
/var/lib/docker/volumes/fantasy-scraper-website_player_data/_data/
```

You can inspect it with:
```bash
docker volume inspect fantasy-scraper-website_player_data
```

## Troubleshooting

### Check if data exists:
```bash
docker run --rm -v player_data:/data alpine ls -lh /data
```

### View sync logs:
```bash
docker-compose run --rm sync
```

### Clear and re-sync:
```bash
docker volume rm fantasy-scraper-website_player_data
docker-compose run --rm sync
```
