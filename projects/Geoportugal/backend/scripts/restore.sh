#!/bin/bash

# GeoPortugal Database Restore Script
# This script restores a PostgreSQL database from backup

set -e

# Configuration
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-geoportugal}"
POSTGRES_USER="${POSTGRES_USER:-geoportugal_user}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
S3_BUCKET="${BACKUP_S3_BUCKET:-}"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] BACKUP_FILE"
    echo ""
    echo "Options:"
    echo "  -s, --s3        Download backup from S3 bucket"
    echo "  -l, --list      List available backups"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 geoportugal_backup_20240101_120000.sql"
    echo "  $0 --s3 geoportugal_backup_20240101_120000.sql"
    echo "  $0 --list"
    exit 1
}

# Function to list backups
list_backups() {
    echo "Local backups:"
    if ls "${BACKUP_DIR}"/geoportugal_backup_*.sql 1> /dev/null 2>&1; then
        ls -lh "${BACKUP_DIR}"/geoportugal_backup_*.sql | awk '{print $9, $5, $6, $7, $8}'
    else
        echo "  No local backups found"
    fi
    
    if [[ -n "${S3_BUCKET}" ]] && command -v aws &> /dev/null; then
        echo ""
        echo "S3 backups:"
        aws s3 ls "s3://${S3_BUCKET}/backups/" --human-readable | grep geoportugal_backup_
    fi
}

# Parse command line arguments
DOWNLOAD_FROM_S3=false
LIST_BACKUPS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--s3)
            DOWNLOAD_FROM_S3=true
            shift
            ;;
        -l|--list)
            LIST_BACKUPS=true
            shift
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            BACKUP_FILE="$1"
            shift
            ;;
    esac
done

# List backups if requested
if [[ "$LIST_BACKUPS" == true ]]; then
    list_backups
    exit 0
fi

# Check if backup file is provided
if [[ -z "${BACKUP_FILE}" ]]; then
    echo "ERROR: Backup file not specified"
    show_usage
fi

BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILE}"

echo "Starting database restore at $(date)"

# Download from S3 if requested
if [[ "$DOWNLOAD_FROM_S3" == true ]]; then
    if [[ -z "${S3_BUCKET}" ]]; then
        echo "ERROR: S3_BUCKET not configured"
        exit 1
    fi
    
    echo "Downloading backup from S3: ${S3_BUCKET}/backups/${BACKUP_FILE}"
    aws s3 cp "s3://${S3_BUCKET}/backups/${BACKUP_FILE}" "${BACKUP_PATH}"
fi

# Check if backup file exists
if [[ ! -f "${BACKUP_PATH}" ]]; then
    echo "ERROR: Backup file not found: ${BACKUP_PATH}"
    echo "Available backups:"
    list_backups
    exit 1
fi

# Confirm restore operation
echo "WARNING: This will replace the current database with the backup!"
echo "Database: ${POSTGRES_DB}"
echo "Backup file: ${BACKUP_FILE}"
echo "Backup size: $(du -h "${BACKUP_PATH}" | cut -f1)"
echo ""
read -p "Are you sure you want to continue? (yes/no): " -r
if [[ ! $REPLY == "yes" ]]; then
    echo "Restore cancelled"
    exit 0
fi

# Create a pre-restore backup
echo "Creating pre-restore backup..."
PRE_RESTORE_BACKUP="geoportugal_pre_restore_$(date +"%Y%m%d_%H%M%S").sql"
pg_dump -h "${POSTGRES_HOST}" \
        -U "${POSTGRES_USER}" \
        -d "${POSTGRES_DB}" \
        --format=custom \
        --compress=9 \
        --file="${BACKUP_DIR}/${PRE_RESTORE_BACKUP}" || echo "Warning: Could not create pre-restore backup"

# Terminate active connections to the database
echo "Terminating active connections..."
psql -h "${POSTGRES_HOST}" \
     -U "${POSTGRES_USER}" \
     -d "postgres" \
     -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${POSTGRES_DB}' AND pid <> pg_backend_pid();" || true

# Drop and recreate database
echo "Dropping and recreating database..."
psql -h "${POSTGRES_HOST}" \
     -U "${POSTGRES_USER}" \
     -d "postgres" \
     -c "DROP DATABASE IF EXISTS ${POSTGRES_DB};"

psql -h "${POSTGRES_HOST}" \
     -U "${POSTGRES_USER}" \
     -d "postgres" \
     -c "CREATE DATABASE ${POSTGRES_DB} WITH OWNER = ${POSTGRES_USER};"

# Enable PostGIS extension
echo "Enabling PostGIS extension..."
psql -h "${POSTGRES_HOST}" \
     -U "${POSTGRES_USER}" \
     -d "${POSTGRES_DB}" \
     -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# Restore database
echo "Restoring database from backup..."
pg_restore -h "${POSTGRES_HOST}" \
           -U "${POSTGRES_USER}" \
           -d "${POSTGRES_DB}" \
           --verbose \
           --clean \
           --if-exists \
           --no-owner \
           --no-privileges \
           "${BACKUP_PATH}"

echo "Database restore completed at $(date)"

# Optional: Send notification
if [[ -n "${BACKUP_WEBHOOK_URL}" ]]; then
    curl -X POST "${BACKUP_WEBHOOK_URL}" \
         -H "Content-Type: application/json" \
         -d "{\"text\":\"🔄 Database restored from backup: ${BACKUP_FILE}\"}" \
         --silent --output /dev/null || echo "Warning: Failed to send restore notification"
fi

echo ""
echo "✅ Restore completed successfully!"
echo "Pre-restore backup saved as: ${PRE_RESTORE_BACKUP}"