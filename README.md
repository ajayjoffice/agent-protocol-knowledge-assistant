# Agent Protocol Knowledge Assistant

A local multi-agent knowledge assistant built with **Python**, **LangChain LCEL**, **Hugging Face**, and **FastAPI**. A planner delegates evidence gathering to a retriever, drafts a response, and asks a critic to review it. The agents exchange explicit, versioned JSON messages, and the workflow returns source references or abstains when retrieval finds no matching evidence.

The project is designed as a compact demonstration of agent roles, message passing, bounded review loops, local model inference, and evaluation. It does not connect to a hosted MCP or A2A service; the protocol is a small in-process example modeled on those ideas.

## What it includes

- **Planner, retriever, and critic roles:** distinct Python components with explicit inputs and outputs.
- **LangChain LCEL:** planner and critic use prompt → model → parser chains.
- **Versioned protocol envelopes:** task and message IDs, sender, recipient, kind, payload, and correlation ID.
- **Local retrieval:** dependency-free BM25 ranking over `.txt` and `.md` files.
- **Bounded review:** the critic can request at most one planner revision.
- **Evidence behavior:** answers include source references; the workflow abstains in code when retrieval returns no documents.
- **Local inference:** Qwen2.5-1.5B-Instruct runs on your machine by default; no model API key or usage credits are required.
- **Two interfaces:** a command-line app and an optional FastAPI service.
- **Evaluation:** a 12-case bundled set for retrieval, citation, critic, and unsupported-question behavior.

## Agent workflow

```text
User
  │ plan
  ▼
Planner ── retrieve request ──▶ Retriever
  ▲                              │
  └──────── evidence reply ──────┘
  │
  ├── draft ──▶ Critic
  │              │
  │◀── critique ─┘
  │
  ├── revise once if requested, then review again
  └── final answer with source references ──▶ User
```

If retrieval returns no documents, the workflow returns an abstention directly without asking the language model to answer from pretrained knowledge. If the model omits a citation for a retrieved answer, the workflow adds the highest-ranked retrieved source as a source reference.

## Agent message protocol

Every handoff uses an `agent-message/1.0` JSON envelope. For example, a planner-to-retriever request looks like this:

```json
{
  "protocol": "agent-message/1.0",
  "task_id": "task-uuid",
  "message_id": "message-uuid",
  "sender": "planner",
  "recipient": "retriever",
  "kind": "retrieve",
  "payload": {"query": "What does the retriever do?"},
  "correlation_id": "plan-message-uuid"
}
```

The in-memory message bus checks the task ID and intended recipient, then records each message in a transcript. Replies use `correlation_id` to point to the request they answer. Run the CLI with `--show-transcript`, or call the API with `"include_trace": true`, to inspect the conversation.

The envelope is transport-neutral, but this demo uses only an in-memory bus. It does not implement the MCP or A2A network protocols.

## Requirements

- Python 3.10 or newer
- About 3.1 GB of disk space for the default model weights, plus space for Python dependencies
- Internet access for the first model download; inference uses the local model afterward
- Apple Silicon MPS, CUDA, or CPU supported through the installed PyTorch build; CPU inference can be slow

The default model is [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). Its weights are downloaded from Hugging Face the first time the local model starts and are cached for later runs.

## Install

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[local,api]'
```

Install only `.[local]` if you plan to use the CLI and evaluation without the API. The model is loaded when a local CLI request or API service starts.

## Run the CLI

Ask a question using the bundled knowledge base:

```bash
python -m agent_system.cli "What does the retriever do?"
```

Show the agent message trace:

```bash
python -m agent_system.cli "What does the retriever do?" --show-transcript
```

Point retrieval at one text/Markdown file or a folder of them:

```bash
python -m agent_system.cli "What do the agents do?" --knowledge ./my-notes
```

The retriever searches folders recursively. It treats each blank-line-separated paragraph as a passage and identifies it in citations as `filename#paragraph-number`. Supported extensions are `.txt` and `.md`; PDF and Word files are not currently parsed.

The CLI uses the local Hugging Face model by default. You can select another compatible model with `--model MODEL_ID` or the `HF_MODEL` environment variable. An optional hosted OpenAI backend is available with `pip install -e '.[openai]'`, `OPENAI_API_KEY`, and `--provider openai`.

The package also defines an `agent-demo` command after installation:

```bash
agent-demo "What is a message-passing protocol useful for?"
```

## Run the API

Start the local service:

```bash
uvicorn agent_system.api:app --host 127.0.0.1 --port 8000
```

Available routes:

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service name and route links |
| `GET` | `/health` | Service status and model ID |
| `POST` | `/ask` | Ask a question; optionally include the protocol trace |
| `GET` | `/docs` | Interactive Swagger UI |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does the retriever do?","include_trace":true}'
```

Example response shape when `include_trace` is omitted or `false`:

```json
{
  "answer": "The retriever gathers evidence.\n\nSource: [knowledge_base.txt#3]",
  "sources": ["knowledge_base.txt#3"],
  "trace": null
}
```

Set `"include_trace": true` in the request body to receive the full ordered message list in `trace`.

Set `HF_MODEL` to select the local model used by the API and `KNOWLEDGE_PATH` to select a `.txt`/`.md` file or folder. The API loads one workflow when it starts and reuses it for requests.

## Evaluate

Run the fast retrieval and abstention-gate evaluation. It makes no model calls:

```bash
python -m agent_system.evaluate
```

Run the full evaluation through the local model and critic:

```bash
python -m agent_system.evaluate --end-to-end
```

The report includes retrieval hit rate at top-k, unsupported-query abstention accuracy, expected-source citation rate, and critic PASS rate, along with per-case results. The full evaluation takes longer because it runs the planner and critic for each supported question. Use `--top-k 4` to change the retrieval cutoff or `--model MODEL_ID` to select another local model.

### Bundled baseline

On the bundled 12-case set, the recorded Qwen2.5-1.5B-Instruct run produced:

- Expected source retrieved within top 4: **8/8 supported cases**
- Unsupported questions with no retrieved evidence: **4/4 cases**
- Expected source included in the final answer: **8/8 supported cases**
- Critic verdict `PASS`: **8/8 supported cases**

These are results for a small, hand-written benchmark and bundled corpus, not a guarantee of general accuracy. Citation coverage includes the workflow's source-reference fallback when the model omits one. The critic is another invocation of the same small model, so its verdict is not an independent ground-truth check. Re-run the evaluation after changing the model, prompts, corpus, or retriever. Update `agent_system/eval_cases.json` with expected source IDs before evaluating a different corpus.

## Project structure

```text
agent_system/
├── agents.py          # Retriever and critic agent roles
├── api.py             # FastAPI routes
├── cli.py             # Command-line entry point
├── eval_cases.json    # Curated 12-case evaluation set
├── evaluate.py        # Retrieval and optional end-to-end evaluation
├── knowledge_base.txt # Bundled example corpus
├── local_model.py     # Local Hugging Face model factory
├── protocol.py        # Message envelope and in-memory router
├── retrieval.py       # Local BM25 retriever
└── workflow.py        # Planner orchestration and review loop
```

## Limitations and design choices

- BM25 is lexical retrieval; semantic embeddings and reranking are not included.
- The benchmark is small and tailored to the bundled corpus.
- The critic is an LLM reviewer, not a formal correctness guarantee.
- Message routing and transcript storage are in memory; state is not persisted across requests or restarts.
- The default 1.5B model is compact for local use, but can miss details or produce weak critiques.
- Retrieval supports plain text and Markdown passages only.
- The sample API is intended for local development and binds to `127.0.0.1` in the documented command; it does not include authentication or production deployment configuration.

