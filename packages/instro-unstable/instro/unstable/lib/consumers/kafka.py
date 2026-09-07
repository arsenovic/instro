"""Kafka consumer implementation.

This mirrors the Kafka publisher implementation so a Kafka-backed publisher can
be paired with a consumer that decodes payloads and forwards them to a handler.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any, Callable

from instro.lib.types import Command, Measurement

logger = logging.getLogger(__name__)

try:
    from kafka import KafkaConsumer as KafkaConsumerClient
except ImportError:  # pragma: no cover - dependency is optional at import time
    KafkaConsumerClient = None

try:
    from instro.lib.consumers.consumer import BaseConsumer
except ImportError:  # pragma: no cover - fallback for unstable package layout
    from instro.unstable.lib.consumers.consumer import BaseConsumer


def _default_key_deserializer(key: bytes | bytearray | str | None) -> Any | None:
    if key is None:
        return None
    if isinstance(key, (bytes, bytearray)):
        return key.decode("utf-8")
    return key


def _default_value_deserializer(value: bytes | bytearray | str | None) -> Any:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value)
        try:
            return json.loads(value.decode("utf-8"))
        except (TypeError, ValueError, UnicodeDecodeError):
            return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


class KafkaConsumer(BaseConsumer):
    """Consumer that decodes Kafka messages and forwards them to a handler."""

    def __init__(
        self,
        topic: str | list[str] | tuple[str, ...],
        *,
        handler: Callable[[Measurement | Command], None] | None = None,
        consumer: KafkaConsumerClient | None = None,
        bootstrap_servers: str | list[str] | tuple[str, ...] | None = None,
        key_deserializer: Callable[[Any], Any] | None = None,
        value_deserializer: Callable[[Any], Any] | None = None,
        consumer_kwargs: dict[str, Any] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        **consumer_options: Any,
    ) -> None:
        super().__init__(on_error=on_error)

        if KafkaConsumerClient is None and consumer is None:
            raise ImportError(
                "kafka-python is required for KafkaConsumer. "
                "Install it via 'pip install kafka-python'."
            )

        self.topic = topic
        self.handler = handler
        self.key_deserializer = key_deserializer or _default_key_deserializer
        self.value_deserializer = value_deserializer or _default_value_deserializer

        if consumer is not None:
            self._consumer = consumer
        else:
            topics = [topic] if isinstance(topic, str) else list(topic)
            client_kwargs = dict(consumer_kwargs or {})
            if bootstrap_servers is not None:
                client_kwargs.setdefault("bootstrap_servers", bootstrap_servers)
            client_kwargs.update(consumer_options)
            self._consumer = KafkaConsumerClient(*topics, **client_kwargs)

    @property
    def consumer(self) -> KafkaConsumerClient:
        return self._consumer

    def _decode_message(self, data: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        if hasattr(data, "value") and hasattr(data, "key"):
            key = self.key_deserializer(data.key)
            value = self.value_deserializer(data.value)
            message_kwargs = dict(kwargs)
            message_kwargs.setdefault("key", key)
            message_kwargs.setdefault("topic", getattr(data, "topic", self.topic))
            message_kwargs.setdefault("partition", getattr(data, "partition", None))
            message_kwargs.setdefault("offset", getattr(data, "offset", None))
            return value, message_kwargs

        if isinstance(data, (bytes, bytearray, str)):
            return self.value_deserializer(data), dict(kwargs)

        return data, dict(kwargs)

    def _consume(self, data: Measurement | Command | Any, **kwargs: Any) -> None:
        if self.handler is None:
            raise ValueError("KafkaConsumer requires a handler to consume records")

        payload, message_kwargs = self._decode_message(data, **kwargs)
        self.handler(payload, **message_kwargs)

    def consume_messages(
        self,
        *,
        timeout_ms: int = 1000,
        auto_commit: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Consume and dispatch all available Kafka messages until the consumer is closed."""
        while not self._closed:
            for message in self._consumer:
                self.consume(message, **kwargs)
                if auto_commit is not None:
                    self._consumer.commit()
                break

    def __iter__(self) -> Iterable[Any]:
        return iter(self._consumer)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if hasattr(self._consumer, "close"):
            self._consumer.close()


__all__ = ["KafkaConsumer"]
