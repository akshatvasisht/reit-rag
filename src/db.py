"""Database connection helpers shared across ingestion and retrieval."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """Yield a psycopg connection with pgvector types registered."""
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Configure it in your environment or .env "
            "before running the application."
        )
    with psycopg.connect(DATABASE_URL, autocommit=False) as conn:
        register_vector(conn)
        yield conn
