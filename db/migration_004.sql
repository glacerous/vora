-- Migration 004: Add carbon calculation metadata and GPS columns to tree_scans table
ALTER TABLE tree_scans ADD COLUMN wood_density_used REAL;
ALTER TABLE tree_scans ADD COLUMN wood_density_source TEXT;
ALTER TABLE tree_scans ADD COLUMN climate_zone_detected TEXT;
ALTER TABLE tree_scans ADD COLUMN formula_used TEXT;
ALTER TABLE tree_scans ADD COLUMN agb_kg REAL;
ALTER TABLE tree_scans ADD COLUMN bgb_kg REAL;
ALTER TABLE tree_scans ADD COLUMN gps_lat REAL;
ALTER TABLE tree_scans ADD COLUMN gps_lon REAL;
