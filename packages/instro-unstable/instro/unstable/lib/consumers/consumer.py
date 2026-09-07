"""Consumer protocol and generic base implementations.

Consumers are the counterpart to publishers: they receive measurement and command
objects emitted by a publisher without imposing any domain-specific behavior.
"""

import abc
import logging
from typing import Callable, Protocol, Any

from instro.lib.types import Command, Measurement

logger = logging.getLogger(__name__)


class Consumer(Protocol):
    """Protocol for objects that receive published data."""

    def consume(self, data: Measurement | Command, **kwargs) -> None: ...

    def close(self) -> None: ...


class BaseConsumer(abc.ABC):
    """Generic consumer base class for publisher-driven workflows."""

    def __init__(self, *, on_error: Callable[[Exception], None] | None = None):
        self._on_error = on_error
        self._closed = False

    def consume(self, data: Measurement | Command, **kwargs) -> None:
        if self._closed:
            logger.warning(
                "Dropping consume request because BaseConsumer is closed (consumer=%s)",
                self.__class__.__name__,
            )
            return

        try:
            self._consume(data, **kwargs)
        except Exception as exc:
            if self._on_error is not None:
                self._on_error(exc)
            else:
                logger.exception("Consumer failed to consume data (consumer=%s)", self.__class__.__name__)

    @abc.abstractmethod
    def _consume(self, data: Measurement | Command, **kwargs) -> None:
        """Implement the actual handling for a given published payload."""

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

    @staticmethod
    def is_measurement_record(data: dict[str, Any]) -> bool:
        """Return True when the given record looks like a Measurement.

        This is a small standalone predicate so callers can change the
        detection logic later without touching reader implementations.
        """
        return "timestamps" in data and "channel_data" in data

    @staticmethod
    def is_command_record(data: dict[str, Any]) -> bool:
        """Return True when the given record looks like a Command."""
        return "timestamp" in data and "channel_data" in data

    @staticmethod
    def record_to_object(data: dict[str, Any]) -> Measurement | Command:
        """Convert a plain dict record into a `Measurement` or `Command`.

        Raises `TypeError` for unsupported records. Readers can call this
        to centralize the JSON<->object conversion logic.
        """
        if BaseConsumer.is_measurement_record(data):
            return Measurement(**data)
        if BaseConsumer.is_command_record(data):
            return Command(**data)
        raise TypeError(f"Unsupported record: {data!r}")


class HandlerConsumer(BaseConsumer):
    """Simple consumer that forwards each payload to a callable handler."""

    def __init__(
        self,
        handler: Callable[[Measurement | Command], None],
        *,
        on_error: Callable[[Exception], None] | None = None,
    ):
        super().__init__(on_error=on_error)
        self.handler = handler

    def _consume(self, data: Measurement | Command, **kwargs) -> None:
        self.handler(data, **kwargs)


__all__ = ["Consumer", "BaseConsumer", "HandlerConsumer"]
