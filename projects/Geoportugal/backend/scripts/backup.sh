#!/bin/bash

# GeoPortugal Database Backup Script
# This script creates backups of the PostgreSQL database and optionally uploads to S3

set -e

# Configuration
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-geoportugal}"
POSTGRES_USER="${POSTGRES_USER:-geoportugal_user}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
S3_BUCKET="${BACKUP_S3_BUCKET:-}"
DATE=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="geoportugal_backup_${DATE}.sql"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILE}"

echo "Starting database backup at $(date)"

# Create backup directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"

# Create database dump
echo "Creating database dump..."
pg_dump -h "${POSTGRES_HOST}" \
        -U "${POSTGRES_USER}" \
        -d "${POSTGRES_DB}" \
        --verbose \
        --no-owner \
        --no-privileges \
        --format=custom \
        --compress=9 \
        --file="${BACKUP_PATH}"

# Verify backup was created
if [[ ! -f "${BACKUP_PATH}" ]]; then
    echo "ERROR: Backup file was not created!"
    exit 1
fi

# Get backup file size
BACKUP_SIZE=$(du -h "${BACKUP_PATH}" | cut -f1)
echo "Backup created successfully: ${BACKUP_FILE} (${BACKUP_SIZE})"

# Upload to S3 if configured
if [[ -n "${S3_BUCKET}" ]]; then
    echo "Uploading backup to S3 bucket: ${S3_BUCKET}"
    aws s3 cp "${BACKUP_PATH}" "s3://${S3_BUCKET}/backups/${BACKUP_FILE}" \
        --storage-class STANDARD_IA
    echo "Backup uploaded to S3 successfully"
fi

# Clean up old local backups
echo "Cleaning up backups older than ${RETENTION_DAYS} days..."
find "${BACKUP_DIR}" -name "geoportugal_backup_*.sql" -mtime +${RETENTION_DAYS} -delete

# Clean up old S3 backups if configured
if [[ -n "${S3_BUCKET}" ]] && command -v aws &> /dev/null; then
    echo "Cleaning up old S3 backups..."
    CUTOFF_DATE=$(date -d "${RETENTION_DAYS} days ago" +"%Y-%m-%d")
    aws s3 ls "s3://${S3_BUCKET}/backups/" | while read -r line; do
        BACKUP_DATE=$(echo $line | awk '{print $1}')
        BACKUP_NAME=$(echo $line | awk '{print $4}')
        if [[ "${BACKUP_DATE}" < "${CUTOFF_DATE}" ]]; then
            echo "Deleting old backup: ${BACKUP_NAME}"
            aws s3 rm "s3://${S3_BUCKET}/backups/${BACKUP_NAME}"
        fi
    done
fi

echo "Backup process completed at $(date)"

# Optional: Send notification (if webhook URL is provided)
if [[ -n "${BACKUP_WEBHOOK_URL}" ]]; then
    curl -X POST "${BACKUP_WEBHOOK_URL}" \
         -H "Content-Type: application/json" \
         -d "{\"text\":\"✅ Database backup completed: ${BACKUP_FILE} (${BACKUP_SIZE})\"}" \
         --silent --output /dev/null || echo "Warning: Failed to send backup notification"
fi