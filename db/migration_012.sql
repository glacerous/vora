-- Migration 012: Add plot_areas table for multiple bounding boxes per plot
CREATE TABLE IF NOT EXISTS plot_areas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plot_id INTEGER NOT NULL,
    name TEXT,
    x1 INTEGER NOT NULL,
    y1 INTEGER NOT NULL,
    x2 INTEGER NOT NULL,
    y2 INTEGER NOT NULL,
    FOREIGN KEY(plot_id) REFERENCES plots(id) ON DELETE CASCADE
);
