"""Planner orchestration for the retriever -> planner -> critic conversation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .agents import CriticAgent, RetrieverAgent, _text
from .protocol import AgentMessage, MessageBus
from .retrieval import LocalRetriever


@dataclass
class WorkflowResult:
    answer: str
    transcript: list[AgentMessage]


class AgentWorkflow:
    """Run one bounded pass: retrieve evidence, draft, critique, and optionally revise."""

    def __init__(self, model: BaseChatModel, corpus_path: Path | None = None) -> None:
        corpus = corpus_path or Path(__file__).with_name("knowledge_base.txt")
        self.model = model
        self.retriever = RetrieverAgent(LocalRetriever(corpus))
        self.critic = CriticAgent(model)
        self.planner_chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are the planner in an evidence-grounded agent team. Answer using "
                        "only the supplied evidence. Do not use outside knowledge. Give one "
                        "direct sentence that answers the question. Copy at least one source ID "
                        "exactly as it appears in the evidence, including brackets, for example "
                        "[knowledge_base.txt#3]. Do not write placeholders such as [source-id]. "
                        "If evidence does not support the answer, say so.",
                    ),
                    (
                        "human",
                        "Question:\n{question}\n\nEvidence:\n{evidence}\n\n"
                        "Critic feedback to address (if any):\n{feedback}\n\n"
                        "Check the requested answer against the exact evidence before replying. "
                        "Keep it to one sentence and use the exact source ID from the evidence.",
                    ),
                ]
            )
            | model
            | StrOutputParser()
        )

    def run(self, question: str) -> WorkflowResult:
        task_id = str(uuid.uuid4())
        bus = MessageBus(task_id)
        plan = bus.send(
            AgentMessage.create(
                task_id=task_id,
                sender="user",
                recipient="planner",
                kind="plan",
                payload={"question": question},
            ),
            to="planner",
        )
        retrieve_request = bus.send(
            AgentMessage.create(
                task_id=task_id,
                sender="planner",
                recipient="retriever",
                kind="retrieve",
                payload={"query": question},
                correlation_id=plan.message_id,
            ),
            to="retriever",
        )
        evidence_message = self.retriever.handle(retrieve_request, bus)
        evidence_docs = evidence_message.payload["documents"]
        evidence = "\n".join(
            f"[{doc['id']}] {doc['text']}" for doc in evidence_docs
        ) or "No matching documents were found in the local knowledge base."

        # Enforce abstention in code when retrieval found nothing. A prompt alone
        # cannot reliably stop a small model from answering from pretrained memory.
        if not evidence_docs:
            answer = "I couldn't find relevant evidence for that question in the local knowledge base."
            final_message = bus.send(
                AgentMessage.create(
                    task_id=task_id,
                    sender="planner",
                    recipient="user",
                    kind="final",
                    payload={"answer": answer, "sources": []},
                    correlation_id=evidence_message.message_id,
                ),
                to="user",
            )
            return WorkflowResult(answer=final_message.payload["answer"], transcript=bus.transcript)

        draft = ""
        feedback = ""
        critique: AgentMessage | None = None
        # One initial draft plus at most one revision keeps the loop bounded.
        for review_round in range(2):
            draft = self._draft(question, evidence, feedback)
            critique = self._critique(question, evidence, draft, bus, task_id)
            feedback = str(critique.payload["feedback"])
            if "VERDICT: REVISE" not in feedback.upper() or review_round == 1:
                break

        # Keep provenance visible even when a small local model omits the citation.
        valid_citation = any(f"[{doc['id']}]" in draft for doc in evidence_docs)
        if not valid_citation:
            draft = f"{draft.rstrip()}\n\nSource: [{evidence_docs[0]['id']}]"

        final_message = bus.send(
            AgentMessage.create(
                task_id=task_id,
                sender="planner",
                recipient="user",
                kind="final",
                payload={"answer": draft, "sources": [doc["id"] for doc in evidence_docs]},
                correlation_id=critique.message_id if critique else None,
            ),
            to="user",
        )
        return WorkflowResult(answer=final_message.payload["answer"], transcript=bus.transcript)

    def _draft(self, question: str, evidence: str, feedback: str = "") -> str:
        return _text(self.planner_chain.invoke(
            {"question": question, "evidence": evidence, "feedback": feedback or "None"}
        )).strip()

    def _critique(
        self, question: str, evidence: str, draft: str, bus: MessageBus, task_id: str
    ) -> AgentMessage:
        request = bus.send(
            AgentMessage.create(
                task_id=task_id,
                sender="planner",
                recipient="critic",
                kind="draft",
                payload={"question": question, "evidence": evidence, "draft": draft},
            ),
            to="critic",
        )
        return self.critic.handle(request, bus)
