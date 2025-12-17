#!/bin/sh
set -e

echo "Container starting..."

mkdir -p "$VECTOR_DB_PATH"


python src/seed_vector_db.py

echo "Seeding step finished"

exec uvicorn src.main:app --host 0.0.0.0 --port 8000
