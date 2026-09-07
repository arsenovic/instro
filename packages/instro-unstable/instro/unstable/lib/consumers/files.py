"""File-based consumers for reading data produced by FilePublisher."""

import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol

import fastavro

from instro.lib.types import Command, Measurement
from instro.unstable.lib.consumers.consumer import BaseConsumer


class FileReader(Protocol):
    """Protocol for file readers that decode payloads into Measurement/Command objects."""

    def read(self) -> Iterable[Measurement | Command]: ...
    def close(self) -> None: ...


class FileConsumer(BaseConsumer):
    """Consumer that reads Measurement/Command records from a file written by FilePublisher."""

    def __init__(
        self,
        file_path: str | Path,
        *,
        format: Literal["json", "jsonl", "csv", "avro"] = "avro",
        handler: Callable[[Measurement | Command], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ):
        super().__init__(on_error=on_error)
        self.file_path = Path(file_path)
        self.format = format
        self.handler = handler
        self._reader: FileReader

        if format == "json":
            self._reader = JsonFileReader(self.file_path)
        elif format == "jsonl":
            self._reader = JsonlFileReader(self.file_path)
        elif format == "csv":
            self._reader = CsvFileReader(self.file_path)
        elif format == "avro":
            self._reader = AvroFileReader(self.file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def read(self) -> list[Measurement | Command]:
        """Read all records from the configured file."""
        return list(self._reader.read())

    def _consume(self, data: Measurement | Command | None = None, **kwargs) -> None:
        """Read every payload from the file and forward each record to the handler."""
        for item in self._reader.read():
            if self.handler is not None:
                self.handler(item, **kwargs)

    def close(self) -> None:
        """Close the underlying file reader."""
        try:
            self._reader.close()
        finally:
            super().close()


class JsonFileReader:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)

    def read(self) -> Iterable[Measurement | Command]:
        if not self.file_path.exists():
            return []
        with open(self.file_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            return [self._to_object(item) for item in payload]
        if payload:
            return [self._to_object(payload)]
        return []

    @staticmethod
    def _to_object(data: dict[str, Any]) -> Measurement | Command:
        return BaseConsumer.record_to_object(data)

    def close(self) -> None:
        pass


class JsonlFileReader:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)

    def read(self) -> Iterable[Measurement | Command]:
        if not self.file_path.exists():
            return []
        records: list[Measurement | Command] = []
        with open(self.file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                records.append(self._to_object(json.loads(line)))
        return records

    @staticmethod
    def _to_object(data: dict[str, Any]) -> Measurement | Command:
        return BaseConsumer.record_to_object(data)

    def close(self) -> None:
        pass


class CsvFileReader:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)

    def read(self) -> Iterable[Measurement | Command]:
        if not self.file_path.exists():
            return []

        records: list[Measurement | Command] = []
        with open(self.file_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("channel") is None:
                    continue
                channel = row["channel"]
                value = _coerce_value(row["value"])
                tags = json.loads(row["tags"]) if row.get("tags") else {}
                if row.get("timestamp") is not None and row.get("channel") is not None:
                    record = Command(
                        timestamp=row["timestamp"],
                        channel_data={channel: value},
                        tags=tags,
                    )
                    records.append(record)
        return records

    def close(self) -> None:
        pass


class AvroFileReader:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)

    def read(self) -> Iterable[Measurement | Command]:
        if not self.file_path.exists():
            return []
        records: list[Measurement | Command] = []
        with open(self.file_path, "rb") as fh:
            for item in fastavro.reader(fh):
                records.append(self._to_object(item))
        return records

    @staticmethod
    def _to_object(data: dict[str, Any]) -> Measurement | Command:
        channel = data["channel"]
        timestamps = data.get("timestamps", [])
        values = data.get("values", [])
        tags = data.get("tags") or {}

        if timestamps:
            return Measurement(
                channel_data={channel: values},
                timestamps=timestamps,
                tags=tags,
            )
        return Command(
            timestamp=timestamps[0] if timestamps else None,
            channel_data={channel: values[0] if values else None},
            tags=tags,
        )

    def close(self) -> None:
        pass


def _coerce_value(value: Any) -> Any:
    if value in ("", "None", "null"):
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


__all__ = ["FileConsumer", "FileReader"]
