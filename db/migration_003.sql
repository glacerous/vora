-- Migration 003: Add species_predictions column to tree_scans table
ALTER TABLE tree_scans ADD COLUMN species_predictions TEXT;
