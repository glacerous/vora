-- Migration 007: Add height-validation metadata for manual override endpoints
-- (manual DBH override, recalculate 2D clicks, adjust-geometry). These columns let
-- the server flag whether a height value passed the automatic is_full_tree_height()
-- validation or was manually supplied by the user (and therefore is NOT auto-verified).

ALTER TABLE tree_scans ADD COLUMN height_validated INTEGER;
ALTER TABLE tree_scans ADD COLUMN height_validation_reason TEXT;
