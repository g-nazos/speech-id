#!/bin/bash
# Runs once on first DB init (empty volume) via /docker-entrypoint-initdb.d.
# Restores the pre-populated dump if present, so the DB is query-ready without
# running the (slow) populate pipeline. The dump omits the HNSW index, which the
# evaluation does not use (it disables index scans), so restore is ~30s.
set -e

DUMP=/dump/voxceleb_db.dump

if [ -f "$DUMP" ]; then
  echo "[restore] Restoring database from $DUMP ..."
  pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --no-owner "$DUMP"
  echo "[restore] Restore complete."
else
  echo "[restore] No dump found at $DUMP; starting with an empty database."
  echo "[restore] Place voxceleb_db.dump in data_base/db_dump/, or populate from"
  echo "[restore] scratch with:  docker compose --profile populate run --rm populate"
fi