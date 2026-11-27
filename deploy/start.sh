#!/bin/sh
# Container entrypoint. Free hosts have no persistent disk, so the first boot copies the demo database
# (MeetUniversity + two stress-test companies) into place; anything added afterwards lives until the next redeploy.
set -e
DB="${SIGNALSTACK_DB:-data/signalstack.db}"
mkdir -p "$(dirname "$DB")"
if [ ! -f "$DB" ] && [ -f deploy/seed.db ]; then
  cp deploy/seed.db "$DB"
  echo "Seeded $DB from deploy/seed.db"
fi
exec uvicorn signalstack.web:app --host 0.0.0.0 --port "${PORT:-8000}"
