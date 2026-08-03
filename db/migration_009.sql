-- Migration 009: Add grid coordinates for visual canvas mapping
ALTER TABLE tree_scans ADD COLUMN grid_position_x INTEGER DEFAULT NULL;
ALTER TABLE tree_scans ADD COLUMN grid_position_y INTEGER DEFAULT NULL;
