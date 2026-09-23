"""Agent roles. Each role receives and returns protocol messages."""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .protocol import AgentMessage, MessageBus
from .retrieval import LocalRetriever


def _text(response: object) -> str:
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        return str(content)
    content = content.strip()
    # Some local text-generation pipelines return the serialized chat prompt
    # along with the completion. Keep only the final assistant turn if present.
    assistant_turns = re.findall(
        r"<\|im_start\|>assistant\s*\n(.*?)(?=<\|im_end\|>|$)",
        content,
        flags=re.DOTALL,
    )
    if assistant_turns:
        content = assistant_turns[-1].strip()
    return content.replace("<|im_end|>", "").strip()


class RetrieverAgent:
    name = "retriever"

    def __init__(self, retriever: LocalRetriever) -> None:
        self.retriever = retriever

    def handle(self, message: AgentMessage, bus: MessageBus) -> AgentMessage:
        if message.kind != "retrieve" or message.recipient != self.name:
            raise ValueError("Retriever accepts only retrieve messages addressed to it")
        query = str(message.payload["query"])
        docs = self.retriever.search(query)
        return bus.send(
            AgentMessage.create(
                task_id=message.task_id,
                sender="retriever",
                recipient="planner",
                kind="evidence",
                payload={"query": query, "documents": docs},
                correlation_id=message.message_id,
            ),
            to="planner",
        )


class CriticAgent:
    name = "critic"

    def __init__(self, model: BaseChatModel) -> None:
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a strict evidence reviewer. Compare the draft with the exact "
                        "question and evidence. Reject role swaps, outside facts, unsupported "
                        "details, or missing/incorrect source IDs. PASS only when the answer "
                        "directly answers the question and its claims are supported by the "
                        "evidence. End with exactly one line: VERDICT: PASS or VERDICT: REVISE. "
                        "If revising, state the correct answer from the evidence and name the "
                        "exact source ID to cite.",
                    ),
                    (
                        "human",
                        "Question:\n{question}\n\nEvidence:\n{evidence}\n\nDraft:\n{draft}",
                    ),
                ]
            )
            | model
            | StrOutputParser()
        )

    def handle(self, message: AgentMessage, bus: MessageBus) -> AgentMessage:
        if message.kind != "draft" or message.recipient != self.name:
            raise ValueError("Critic accepts only draft messages addressed to it")
        feedback = _text(self.chain.invoke(
            {
                "question": message.payload["question"],
                "evidence": message.payload["evidence"],
                "draft": message.payload["draft"],
            }
        ))
        return bus.send(
            AgentMessage.create(
                task_id=message.task_id,
                sender="critic",
                recipient="planner",
                kind="critique",
                payload={"feedback": feedback},
                correlation_id=message.message_id,
            ),
            to="planner",
        )
