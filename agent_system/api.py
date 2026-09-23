"""Minimal REST interface for the local planner/retriever/critic workflow."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from .local_model import create_local_chat_model
from .workflow import AgentWorkflow


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    include_trace: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    trace: list[dict[str, Any]] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_id = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    knowledge = Path(os.getenv("KNOWLEDGE_PATH", str(Path(__file__).with_name("knowledge_base.txt"))))
    app.state.model_id = model_id
    app.state.workflow = AgentWorkflow(create_local_chat_model(model_id), corpus_path=knowledge)
    yield


app = FastAPI(
    title="Agent Protocol Knowledge Assistant",
    description="Local planner, retriever, and critic agents connected by a versioned message protocol.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def index() -> dict[str, str]:
    return {
        "name": "Agent Protocol Knowledge Assistant",
        "health": "/health",
        "ask": "POST /ask",
        "interactive_docs": "/docs",
    }


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "provider": "local", "model": request.app.state.model_id}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    result = request.app.state.workflow.run(payload.question)
    final = result.transcript[-1]
    trace = (
        [json.loads(message.to_json()) for message in result.transcript]
        if payload.include_trace
        else None
    )
    return AskResponse(
        answer=result.answer,
        sources=list(final.payload.get("sources", [])),
        trace=trace,
    )
