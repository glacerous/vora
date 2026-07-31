-- Delete duplicate garbage records for POHON-6144
DELETE FROM tree_scans 
WHERE tree_code = 'POHON-6144' 
  AND id IN (26, 27, 28, 29);
