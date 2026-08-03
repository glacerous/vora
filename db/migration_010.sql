-- Migration 010: Add target_co2e_kg optional field to plots
ALTER TABLE plots ADD COLUMN target_co2e_kg REAL DEFAULT NULL;
