-- Migration 002: Add geometry_3d column to tree_scans table
ALTER TABLE tree_scans ADD COLUMN geometry_3d TEXT;
