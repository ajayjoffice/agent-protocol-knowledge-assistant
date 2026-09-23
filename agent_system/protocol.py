"""Small, transport-neutral message envelope for agent-to-agent handoffs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal
import json
import uuid

AgentName = Literal["user", "planner", "retriever", "critic"]
MessageKind = Literal["plan", "retrieve", "evidence", "draft", "critique", "final"]


@dataclass(frozen=True)
class AgentMessage:
    """An immutable message; payload is JSON-compatible for easy transport/logging."""

    task_id: str
    message_id: str
    sender: AgentName
    recipient: AgentName
    kind: MessageKind
    payload: dict[str, Any]
    correlation_id: str | None = None
    protocol: str = "agent-message/1.0"

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        sender: AgentName,
        recipient: AgentName,
        kind: MessageKind,
        payload: dict[str, Any],
        correlation_id: str | None = None,
    ) -> "AgentMessage":
        return cls(
            task_id=task_id,
            message_id=str(uuid.uuid4()),
            sender=sender,
            recipient=recipient,
            kind=kind,
            payload=payload,
            correlation_id=correlation_id,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "AgentMessage":
        data = json.loads(raw)
        if data.get("protocol") != "agent-message/1.0":
            raise ValueError("Unsupported message protocol")
        return cls(**data)


class MessageBus:
    """In-memory router that validates recipient and task correlation."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.transcript: list[AgentMessage] = []

    def send(self, message: AgentMessage, *, to: AgentName) -> AgentMessage:
        if message.task_id != self.task_id:
            raise ValueError("Message belongs to a different task")
        if message.recipient != to:
            raise ValueError(f"Message addressed to {message.recipient}, not {to}")
        self.transcript.append(message)
        return message
