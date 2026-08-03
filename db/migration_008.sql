-- Migration 008: Add tables for users, sessions, plots, and link scans
-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    display_name TEXT,
    is_demo_account INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

-- Create sessions table
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create plots table
CREATE TABLE IF NOT EXISTS plots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plot_code TEXT NOT NULL UNIQUE,
    owner_user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    privacy TEXT NOT NULL DEFAULT 'private',
    gps_centroid_lat REAL,
    gps_centroid_lon REAL,
    session_active INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(owner_user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Add relation columns to tree_scans
ALTER TABLE tree_scans ADD COLUMN plot_id INTEGER;
ALTER TABLE tree_scans ADD COLUMN claimed_by_user_id INTEGER;
