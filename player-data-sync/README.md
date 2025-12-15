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
docker volume rm fantasy-scraper-website_darko_data
docker volume rm fantasy-scraper-website_team_pace_data
docker volume rm fantasy-scraper-website_player_season_avgs

# 3. Download and generate everything
docker-compose run --rm -e SYNC_MODE=both sync

# 4. Restart the web app
docker-compose up -d
```

## Viewing Data

To inspect the data inside the Docker volumes without entering a container:

### List files in a volume
```bash
# List player data
docker run --rm -v fantasy-scraper-website_player_data:/data alpine ls -lh /data

# List DARKO data
docker run --rm -v fantasy-scraper-website_darko_data:/data alpine ls -lh /data

# List PMF data
docker run --rm -v fantasy-scraper-website_player_data_pmf:/data alpine ls -lh /data
```

### View file content
To view a specific file (e.g., a player's JSON):
```bash
docker run --rm -v fantasy-scraper-website_player_data:/data alpine cat /data/nikola_jokic.json
```

## Troubleshooting

### Check if data exists:
```bash
docker run --rm -v fantasy-scraper-website_player_data:/data alpine ls -lh /data
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
