from __future__ import annotations

import abc
import time
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MeasurementLike(Protocol):
    """Small structural protocol for streaming measurement payloads.

    The real project uses a dataclass named Measurement, but the sink is designed
    to accept any object exposing the same fields: a timestamp series and a
    mapping of channel names to equal-length numeric arrays.
    """

    timestamps: Sequence[str | datetime | date | int | float]
    channel_data: Mapping[str, Sequence[float | int]]


class BaseSink(abc.ABC):
    """Abstract sink base class implementing the common buffer lifecycle."""

    def __init__(self, *, batch_size: int = 5000, max_interval_sec: float = 5.0) -> None:
        self.batch_size = max(1, int(batch_size))
        self.max_interval_sec = float(max_interval_sec)
        self._buffer: list[tuple[datetime, str, float]] = []
        self._last_flush_ts = time.monotonic()

    @abc.abstractmethod
    def _ensure_schema(self) -> None:
        """Create destination table or collection if it does not exist."""

    @abc.abstractmethod
    def _connect(self) -> Any:
        """Create/return the underlying database client connection."""

    @abc.abstractmethod
    def _bulk_insert(self, rows: list[tuple[datetime, str, float]]) -> None:
        """Persist a batch of rows in a single native bulk operation."""

    def add(self, item: MeasurementLike) -> None:
        """Normalize a measurement-like object into flat rows and append to the buffer."""
        timestamps = list(item.timestamps)
        if not timestamps:
            return

        if not hasattr(item, "channel_data"):
            raise TypeError(f"Measurement-like object missing channel_data: {item!r}")

        channel_data = item.channel_data
        if not isinstance(channel_data, Mapping):
            raise TypeError("channel_data must be a mapping of channel -> sequence of values")

        for channel_name, values in channel_data.items():
            value_list = list(values)
            if len(value_list) != len(timestamps):
                raise ValueError(
                    "Channel data length mismatch: "
                    f"channel={channel_name!r}, timestamps={len(timestamps)}, values={len(value_list)}"
                )

            for ts, value in zip(timestamps, value_list, strict=True):
                self._buffer.append((self._normalize_timestamp(ts), str(channel_name), float(value)))

    def needs_flush(self) -> bool:
        """Return True when the buffer is large enough or too old to flush."""
        elapsed = time.monotonic() - self._last_flush_ts
        return len(self._buffer) >= self.batch_size or elapsed >= self.max_interval_sec

    def flush(self) -> int:
        """Bulk-insert buffered rows, clear the buffer, and return the row count."""
        if not self._buffer:
            return 0

        rows = self._buffer
        try:
            self._bulk_insert(rows)
        except Exception:
            # Let the caller handle logging or retries. The buffer is preserved so
            # the data can be retried after a transient failure.
            raise
        else:
            self._buffer.clear()
            self._last_flush_ts = time.monotonic()
            return len(rows)

    def close(self) -> None:
        """Flush any remaining rows and release the database connection."""
        if self._buffer:
            self.flush()
        self._close_connection()

    @staticmethod
    def _normalize_timestamp(value: str | datetime | date | int | float) -> datetime:
        """Coerce timestamps to timezone-aware UTC datetime objects."""
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, date):
            dt = datetime.combine(value, datetime.min.time())
        elif isinstance(value, (int, float)):
            # Support epoch timestamps as a convenience for non-ISO sources.
            ts = float(value)
            if abs(ts) > 1_000_000_000_000:
                ts /= 1_000_000_000
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        elif isinstance(value, str):
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
        else:
            raise TypeError(f"Unsupported timestamp type: {type(value)!r}")

        return dt.astimezone(timezone.utc) if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

    @abc.abstractmethod
    def _close_connection(self) -> None:
        """Close all open database connections."""



