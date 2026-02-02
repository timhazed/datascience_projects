-- PostgreSQL initialization script for GeoPortugal database

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE geoportugal'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'geoportugal')\gexec

-- Connect to the database
\c geoportugal;

-- Enable PostGIS extension for geospatial operations (optional)
-- CREATE EXTENSION IF NOT EXISTS postgis;

-- Create trigram extension for better text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create indexes for better performance (will be created by Alembic migrations)
-- These are just examples of what could be added

-- Full-text search index on location names
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_districts_name_gin 
--     ON districts USING gin(name gin_trgm_ops);
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_municipalities_name_gin 
--     ON municipalities USING gin(name gin_trgm_ops);  
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_localities_name_gin 
--     ON localities USING gin(name gin_trgm_ops);

-- Geospatial index for proximity searches
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_localities_geom 
--     ON localities USING gist(ST_Point(longitude, latitude));