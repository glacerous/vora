-- Migration 013: Add inlier_ratio column to tree_scans
ALTER TABLE tree_scans ADD COLUMN inlier_ratio REAL DEFAULT NULL;
