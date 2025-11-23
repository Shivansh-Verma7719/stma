#!/bin/bash

# Exit on error
set -e

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

# Path to .env file
ENV_FILE="$PROJECT_ROOT/pipelines/.env"

# Check if .env exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

# Export variables from .env
set -a
source "$ENV_FILE"
set +a

# Check if PGPASSWORD is set, if not use DB_PASSWORD
if [ -z "$PGPASSWORD" ] && [ -n "$DB_PASSWORD" ]; then
    export PGPASSWORD="$DB_PASSWORD"
fi

# Path to schema file
SCHEMA_FILE="$PROJECT_ROOT/infra/db/schema.sql"

if [ ! -f "$SCHEMA_FILE" ]; then
    echo "Error: Schema file not found at $SCHEMA_FILE"
    exit 1
fi

echo "Applying schema to database $DB_NAME at $DB_HOST..."


psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$SCHEMA_FILE"

echo "Schema applied successfully!"
