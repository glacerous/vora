-- Migration 006: Delete old failed scan records with NULL dbh_cm
DELETE FROM tree_scans WHERE id IN (34, 36);
