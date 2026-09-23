from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .local_model import create_local_chat_model
from .workflow import AgentWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the planner/retriever/critic demo")
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--show-transcript", action="store_true", help="Print protocol messages")
    parser.add_argument(
        "--provider", choices=("local", "openai"), default=os.getenv("LLM_PROVIDER", "local"),
        help="Model backend (local Hugging Face by default)",
    )
    parser.add_argument("--model", help="Model ID; defaults based on --provider")
    parser.add_argument(
        "--knowledge",
        type=Path,
        help="A .txt/.md file or folder of local notes (defaults to the bundled knowledge base)",
    )
    args = parser.parse_args()
    question = args.question or input("Question: ").strip()
    if not question:
        parser.error("provide a question")
    if args.provider == "local":
        model_id = args.model or os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        try:
            model = create_local_chat_model(model_id)
        except RuntimeError as exc:
            print(
                str(exc),
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
    else:
        if not os.getenv("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY to use the OpenAI provider.", file=sys.stderr)
            raise SystemExit(2)
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            print("Install the OpenAI integration with: pip install -e '.[openai]'", file=sys.stderr)
            raise SystemExit(2) from exc
        model = ChatOpenAI(
            model=args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0
        )

    workflow = AgentWorkflow(model, corpus_path=args.knowledge)
    result = workflow.run(question)
    print(result.answer)
    if args.show_transcript:
        print("\nProtocol transcript:")
        for message in result.transcript:
            print(message.to_json())


if __name__ == "__main__":
    main()
