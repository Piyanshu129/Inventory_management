-- Inventory Management Database Schema

CREATE TABLE IF NOT EXISTS products (
    product_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    category     TEXT NOT NULL,
    stock        INTEGER NOT NULL DEFAULT 0,
    reorder_level INTEGER NOT NULL DEFAULT 10,
    price        REAL NOT NULL DEFAULT 0.0,
    description  TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_stock ON products(stock);
