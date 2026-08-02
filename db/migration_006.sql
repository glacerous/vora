-- Migration 006: Add accuracy/quality metadata columns to tree_scans table
-- for the carbon-accuracy improvements (scale calibration status, height usage,
-- root-to-shoot ratio, uncertainty interval, and reconstruction quality gate).

ALTER TABLE tree_scans ADD COLUMN scale_status TEXT;
ALTER TABLE tree_scans ADD COLUMN scale_factor_used REAL;
ALTER TABLE tree_scans ADD COLUMN calibration_source TEXT;
ALTER TABLE tree_scans ADD COLUMN height_used TEXT;
ALTER TABLE tree_scans ADD COLUMN total_height_used_m REAL;
ALTER TABLE tree_scans ADD COLUMN segment_height_m REAL;
ALTER TABLE tree_scans ADD COLUMN height_fallback_reason TEXT;
ALTER TABLE tree_scans ADD COLUMN quality_status TEXT;
ALTER TABLE tree_scans ADD COLUMN root_to_shoot_ratio REAL;
ALTER TABLE tree_scans ADD COLUMN co2e_uncertainty_pct REAL;
ALTER TABLE tree_scans ADD COLUMN co2e_low_kg REAL;
ALTER TABLE tree_scans ADD COLUMN co2e_high_kg REAL;

-- Migration 006: Delete old failed scan records with NULL dbh_cm
DELETE FROM tree_scans WHERE id IN (34, 36);
