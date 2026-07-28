CREATE TABLE IF NOT EXISTS tree_scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tree_code TEXT NOT NULL,
    scan_date TEXT NOT NULL,
    dbh_cm REAL,
    tinggi_m REAL,
    biomassa_kg REAL,
    karbon_kg REAL,
    co2e_kg REAL,
    splat_file_url TEXT,
    confidence_note TEXT,
    thumbnail_url TEXT
);
