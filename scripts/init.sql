-- Runs once when the postgres container initialises its data directory.
-- Schema migrations (tables + indexes) live in scripts/migrate.py.

CREATE EXTENSION IF NOT EXISTS vector;
