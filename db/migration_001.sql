-- Migration 001: Add thumbnail_url column to tree_scans table
ALTER TABLE tree_scans ADD COLUMN thumbnail_url TEXT;
