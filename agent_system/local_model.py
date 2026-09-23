"""Factory for the default, fully local LangChain chat model."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def create_local_chat_model(model_id: str | None = None) -> BaseChatModel:
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    except ImportError as exc:
        raise RuntimeError("Install local dependencies with: pip install -e '.[local]'") from exc

    pipeline = HuggingFacePipeline.from_model_id(
        model_id=model_id or "Qwen/Qwen2.5-1.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 512,
            "return_full_text": False,
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.9,
        },
        model_kwargs={"dtype": "auto"},
    )
    return ChatHuggingFace(llm=pipeline)
