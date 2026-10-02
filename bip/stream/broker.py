"""A minimal message-broker interface with in-memory and Kafka implementations.

The in-memory broker runs tests and demos with no infrastructure; the Kafka
broker runs the same producer and consumer code against a real cluster
(see docker-compose.yml, which starts Redpanda, a Kafka-compatible broker).
Messages are JSON dicts; keys are cell IDs, so all of a cell's cycles land on
the same partition and arrive in order.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Iterator, Protocol


class Broker(Protocol):
    def send(self, topic: str, key: str, value: dict) -> None: ...
    def consume(self, topic: str, max_messages: int | None = None) -> Iterator[dict]: ...
    def flush(self) -> None: ...


class InMemoryBroker:
    def __init__(self):
        self.topics: dict[str, deque] = defaultdict(deque)

    def send(self, topic: str, key: str, value: dict) -> None:
        # round-trip through JSON so tests catch anything that wouldn't serialize for Kafka
        self.topics[topic].append(json.loads(json.dumps({"key": key, **value})))

    def consume(self, topic: str, max_messages: int | None = None) -> Iterator[dict]:
        q = self.topics[topic]
        n = 0
        while q and (max_messages is None or n < max_messages):
            yield q.popleft()
            n += 1

    def flush(self) -> None:
        pass


class KafkaBroker:
    def __init__(self, bootstrap: str = "localhost:9092", group_id: str = "bip-consumer"):
        from kafka import KafkaConsumer, KafkaProducer  # pip install ".[stream]"

        self._consumer_cls = KafkaConsumer
        self.bootstrap, self.group_id = bootstrap, group_id
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap,
            key_serializer=lambda k: k.encode(),
            value_serializer=lambda v: json.dumps(v).encode(),
            acks="all",
        )

    def send(self, topic: str, key: str, value: dict) -> None:
        self.producer.send(topic, key=key, value={"key": key, **value})

    def consume(self, topic: str, max_messages: int | None = None) -> Iterator[dict]:
        consumer = self._consumer_cls(
            topic, bootstrap_servers=self.bootstrap, group_id=self.group_id,
            auto_offset_reset="earliest", enable_auto_commit=True, consumer_timeout_ms=5000,
            value_deserializer=lambda b: json.loads(b.decode()),
        )
        for n, msg in enumerate(consumer, 1):
            yield msg.value
            if max_messages is not None and n >= max_messages:
                break
        consumer.close()

    def flush(self) -> None:
        self.producer.flush()
