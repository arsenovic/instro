"""Kafka publisher implementation.

This uses the open-source kafka-python client rather than the Confluent client
so the implementation can be swapped with a different driver later without
changing the public publisher interface.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from instro.lib.types import Command, Measurement

logger = logging.getLogger(__name__)

try:
    from kafka import KafkaProducer as KafkaProducerClient
except ImportError:  # pragma: no cover - dependency is optional at import time
    KafkaProducerClient = None


def _default_key_serializer(key: Any) -> bytes | None:
    if key is None:
        return None
    if isinstance(key, (bytes, bytearray)):
        return bytes(key)
    return str(key).encode("utf-8")


def _default_value_serializer(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value, default=str).encode("utf-8")


class KafkaPublisher:
    """Publish measurement/command objects to a Kafka topic.

    The implementation is intentionally written against a generic producer
    interface so a different Kafka client can be swapped in without changing the
    publisher API used elsewhere in the package.
    """

    def __init__(
        self,
        topic: str,
        bootstrap_servers: str | list[str] | tuple[str, ...] | None = None,
        *,
        producer: KafkaProducerClient | None = None,
        key: Any | None = None,
        key_serializer: Callable[[Any], bytes | None] | None = None,
        value_serializer: Callable[[Any], bytes] | None = None,
        producer_kwargs: dict[str, Any] | None = None,
        **send_kwargs: Any,
    ) -> None:
        if KafkaProducerClient is None and producer is None:
            raise ImportError(
                "kafka-python is required for KafkaPublisher. "
                "Install it via 'pip install kafka-python'."
            )

        self.topic = topic
        self.key = key
        self.key_serializer = key_serializer or _default_key_serializer
        self.value_serializer = value_serializer or _default_value_serializer
        self.send_kwargs = dict(send_kwargs)

        if producer is not None:
            self._producer = producer
        else:
            client_kwargs = dict(producer_kwargs or {})
            if bootstrap_servers is not None:
                client_kwargs.setdefault("bootstrap_servers", bootstrap_servers)
            self._producer = KafkaProducerClient(**client_kwargs)

    def publish(self, data: Measurement | Command, **kwargs) -> None:
        """Submit data to the configured Kafka topic."""
        payload = self.value_serializer(data)
        send_options = dict(self.send_kwargs)
        send_options.update(kwargs)

        key = send_options.pop("key", self.key)
        topic = send_options.pop("topic", self.topic)

        self._producer.send(
            topic,
            key=self.key_serializer(key),
            value=payload,
            **send_options,
        )

    def close(self) -> None:
        if hasattr(self._producer, "flush"):
            self._producer.flush()
        if hasattr(self._producer, "close"):
            self._producer.close()


__all__ = ["KafkaPublisher"]
