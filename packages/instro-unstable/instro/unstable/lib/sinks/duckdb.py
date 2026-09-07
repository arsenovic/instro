from __future__ import annotations

from datetime import datetime
from typing import Any
import duckdb
from .sink import BaseSink


class DuckDBSink(BaseSink):
    """DuckDB-backed sink for streaming measurement rows.

    Schema:
        CREATE TABLE measurements (
            time TIMESTAMP,
            channel_name VARCHAR,
            value DOUBLE
        );
    """

    def __init__(
        self,
        connection_string: str,
        *,
        batch_size: int = 5000,
        max_interval_sec: float = 5.0,
    ) -> None:
        super().__init__(batch_size=batch_size, max_interval_sec=max_interval_sec)
        self.connection_string = connection_string
        self._conn = self._connect()
        self._ensure_schema()

    def _connect(self) -> Any:
        return duckdb.connect(self.connection_string)

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS measurements (
                time TIMESTAMP,
                channel_name VARCHAR,
                value DOUBLE
            )
            """
        )

    def _bulk_insert(self, rows: list[tuple[datetime, str, float]]) -> None:
        """Bulk insert rows using DuckDB's native parameterized executemany."""
        if not rows:
            return

        try:
            self._conn.execute("BEGIN")
            self._conn.executemany(
                "INSERT INTO measurements (time, channel_name, value) VALUES (?, ?, ?)",
                [(ts, ch, value) for ts, ch, value in rows],
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.rollback()
            raise

    def _close_connection(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def __enter__(self) -> "DuckDBSink":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
