"""Offline retrieval and abstention-gate evaluation; no LLM call is needed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .local_model import create_local_chat_model
from .retrieval import LocalRetriever
from .workflow import AgentWorkflow


def evaluate(
    knowledge: Path, top_k: int, *, end_to_end: bool = False, model_id: str | None = None
) -> dict[str, object]:
    cases_path = Path(__file__).with_name("eval_cases.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    retriever = LocalRetriever(knowledge, limit=top_k)
    rows = []
    answerable = [case for case in cases if case["expected_source"] is not None]
    unsupported = [case for case in cases if case["expected_source"] is None]

    for case in cases:
        results = retriever.search(case["question"])
        ids = [item["id"] for item in results]
        expected = case["expected_source"]
        passed = expected in ids if expected else not results
        rows.append(
            {
                "id": case["id"],
                "pass": passed,
                "expected_source": expected,
                "retrieved": ids,
            }
        )

    answerable_rows = [row for row in rows if row["expected_source"] is not None]
    unsupported_rows = [row for row in rows if row["expected_source"] is None]
    hit_rate = sum(bool(row["pass"]) for row in answerable_rows) / len(answerable)
    abstention_accuracy = sum(bool(row["pass"]) for row in unsupported_rows) / len(unsupported)
    overall = sum(bool(row["pass"]) for row in rows) / len(rows)
    report: dict[str, object] = {
        "top_k": top_k,
        "cases": len(rows),
        "retrieval_hit_rate": round(hit_rate, 3),
        "unsupported_query_abstention_accuracy": round(abstention_accuracy, 3),
        "overall_case_pass_rate": round(overall, 3),
        "results": rows,
    }
    if end_to_end:
        workflow = AgentWorkflow(create_local_chat_model(model_id), corpus_path=knowledge)
        answerable_passes = []
        unsupported_passes = []
        review_passes = []
        end_to_end_rows = []
        for case in cases:
            result = workflow.run(case["question"])
            expected = case["expected_source"]
            answer = result.answer
            critiques = [
                message for message in result.transcript
                if message.sender == "critic" and message.kind == "critique"
            ]
            if expected:
                grounded = f"[{expected}]" in answer
                answerable_passes.append(grounded)
                final_feedback = str(critiques[-1].payload["feedback"]) if critiques else ""
                review_passes.append(bool(critiques) and "VERDICT: PASS" in final_feedback.upper())
                passed = grounded
            else:
                abstained = "couldn't find relevant evidence" in answer.lower()
                unsupported_passes.append(abstained)
                passed = abstained
            end_to_end_rows.append(
                {
                    "id": case["id"],
                    "pass": passed,
                    "answer": answer,
                    "critic_rounds": len(critiques),
                }
            )
        report["end_to_end"] = {
            "model": model_id or "Qwen/Qwen2.5-1.5B-Instruct",
            "expected_source_citation_rate": round(
                sum(answerable_passes) / len(answerable_passes), 3
            ),
            "unsupported_abstention_accuracy": round(
                sum(unsupported_passes) / len(unsupported_passes), 3
            ),
            "critic_pass_rate": round(sum(review_passes) / len(review_passes), 3),
            "results": end_to_end_rows,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and no-evidence abstention")
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=Path(__file__).with_name("knowledge_base.txt"),
        help="Knowledge file or folder (defaults to the bundled example corpus)",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Number of passages to retrieve")
    parser.add_argument(
        "--end-to-end",
        action="store_true",
        help="Also run the local model and score citations, abstentions, and critic reviews",
    )
    parser.add_argument("--model", help="Local Hugging Face model ID for --end-to-end")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    report = evaluate(args.knowledge, args.top_k, end_to_end=args.end_to_end, model_id=args.model)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
