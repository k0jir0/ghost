# Ghost — AI Model Context & Training Platform

Ghost is an intelligent ML training and inference platform that combines **PyTorch** and **TensorFlow** backends with the **Model Context Protocol (MCP)** for context-aware AI interactions and **Ollama** for local LLM-powered assistance. The current codebase is organized around explicit planning, dataset resolution, orchestration, and tool-catalog layers so the autonomous agent and MCP server share the same training primitives.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         GHOST PLATFORM                            │
├────────────────────────────────────────────────────────────────────┤
│  Agent + MCP Entry Points                                         │
│  ├── TrainingAgent (TASKS.json queue, AGENT.json state)           │
│  └── GhostMCPServer (tool transport + dispatch)                   │
├────────────────────────────────────────────────────────────────────┤
│  Data + Experiment Control Plane                                  │
│  ├── datasets.py / data_loading.py / ingestion-ready registry     │
│  ├── dataset_registry.py / data_validation.py                     │
│  ├── orchestration.py / experiment_tracking.py / run_store.py     │
│  ├── evaluation.py / model_registry.py / audit.py                 │
│  └── task_queue.py / workflows.py / scheduler.py                  │
├────────────────────────────────────────────────────────────────────┤
│  Serving + Production Signals                                     │
│  ├── inference.py / serving.py / prediction_schemas.py            │
│  ├── observability.py / drift.py / alerts.py                      │
│  └── auth.py / environment.py / deploy/                           │
├────────────────────────────────────────────────────────────────────┤
│  Runtime + Backends                                               │
│  ├── context.py / training.py                                     │
│  ├── pytorch_ops.py                                               │
│  └── tensorflow_ops.py                                            │
├────────────────────────────────────────────────────────────────────┤
│  Ollama Layer (ollama_client.py)                                  │
│  ├── Local LLM inference                                          │
│  ├── Model recommendation engine                                  │
│  └── Training analysis from recorded metrics                      │
└────────────────────────────────────────────────────────────────────┘
```

## Features

- **Dual Framework Support** — Seamless PyTorch and TensorFlow integration with a unified API
- **MCP Protocol** — Standardized tool interface for AI model interactions via `GhostMCPServer`
- **Local LLM** — Ollama integration for private, offline inference and model recommendations
- **Autonomous Training Agent** — Watches `TASKS.json` by default, exposes queue CRUD through MCP tools, and persists machine state in `AGENT.json`
- **Recommendation-Driven Planning** — The agent uses Ollama recommendations to select architecture and training hyperparameters before execution
- **Real Dataset Runtime** — CIFAR-10, MNIST, and IMDB-review datasets now flow through a shared cached loader and batch provider instead of synthetic-only scaffolding
- **Governed Ingestion Interfaces** — `ingestion.py` now supports file-backed dataset bundles directly and object-store-backed bundles through a pluggable fetch callback, so external dataset specs can be versioned and loaded without hardcoding them into the runtime
- **Shared Service Layer** — Planning, dataset resolution, orchestration, and tool-catalog modules keep the agent and MCP server aligned
- **Experiment Tracking** — Searchable persisted run records and artifact lineage now sit alongside orchestration metadata
- **Registry + Gates** — Evaluation thresholds, draft/staging/production registry stages, approval metadata, and audit logging gate model promotion
- **Serving Surface** — Registry-backed online and batch inference are available through `inference.py`, with an optional FastAPI app factory in `serving.py`
- **Production Signals** — Prediction observability, drift reports, alerts, and drift-triggered retraining workflows are now first-class services
- **Environment Isolation** — `environment.py`, `auth.py`, and `deploy/` define dev/staging/production separation and lightweight access-control primitives
- **Explicit Demo Mode** — Synthetic training and evaluation data is disabled by default and must be opted into for scaffold/demo runs
- **Health Monitoring** — GPU, memory, and cache monitoring with configurable thresholds
- **Graceful Degradation** — Memory pressure automatically reduces training batch size before the run escalates
- **Type-Safe Config** — Pydantic-based `GhostConfig` with `.env` file support

## Recent Progress

- Dataset manifests, validation reports, searchable experiment runs, and artifact lineage are now persisted under the shared metadata store.
- File-backed and object-store-backed dataset ingestion now feed external `.npz` dataset bundles through the same governed manifest and validation path as built-in datasets.
- Models can now be evaluated, registered, promoted, rejected, and served from registry-managed versions.
- Prediction traffic now records observability events, drift summaries, alerts, and drift-triggered retraining workflows.
- The current regression baseline is the full green suite for the completed roadmap checkpoint.

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, for GPU training)
- [Ollama](https://ollama.ai) installed locally

### Installation

```bash
# Clone the repo
git clone https://github.com/k0jir0/ghost.git
cd ghost

# Install the package
pip install -e .

# Or install with development tooling
pip install -e ".[dev]"

# Or include the optional FastAPI serving surface
pip install -e ".[serve]"

# (Optional) Pull Ollama models
ollama pull llama3
ollama pull mistral
```

### Configuration

```bash
# Create your .env file
cp .env.example .env   # PowerShell: Copy-Item .env.example .env

# Key settings:
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_MODEL=llama3
# AI_BACKEND=ollama            # only supported AI backend today
# TRAINING_BACKEND=auto          # pytorch | tensorflow | auto
# GPU_ENABLED=true
# LOG_LEVEL=INFO
# MODEL_CACHE_DIR=./models
# DATA_CACHE_DIR=./data
# TASK_QUEUE_FILE=./TASKS.json
# AGENT_STATE_FILE=./AGENT.json
# ALLOW_SYNTHETIC_DATA=false    # set true only for demo/scaffold runs
```

### Running

```bash
# Start the MCP server
python -m ghost.mcp_server

# Or use the installed console script
ghost

# Start the autonomous training agent
python -m agents.training_agent

# Or use the installed console script
ghost-agent

# Startup scripts remain available for the server flow
./start.sh server   # Linux/macOS
./start.bat server  # Windows
```

## Project Structure

```
ghost/
├── src/
│   └── ghost/
│       ├── __init__.py           # Public package exports
│       ├── config.py             # Pydantic-based configuration (GhostConfig)
│       ├── context.py            # Model context & state management
│       ├── data_loading.py       # Real dataset loading and batch provisioning
│       ├── datasets.py           # Dataset resolution and demo-mode policy
│       ├── ingestion.py          # Filesystem and object-store dataset ingestion interfaces
│       ├── health_monitor.py     # Resource monitoring and adaptive health checks
│       ├── experiment_tracking.py# Searchable experiment lineage materialization
│       ├── run_store.py          # Persistent experiment and artifact records
│       ├── evaluation.py         # Threshold-based model evaluation gates
│       ├── model_registry.py     # Versioned model registry and promotion states
│       ├── inference.py          # Registry-backed online and batch prediction service
│       ├── serving.py            # Optional FastAPI serving app factory
│       ├── prediction_schemas.py # Prediction request/response schemas
│       ├── observability.py      # Prediction event logging and aggregate metrics
│       ├── drift.py              # Drift reports from production prediction history
│       ├── alerts.py             # Alert derivation from observability + drift signals
│       ├── retraining.py         # Retraining request creation from operational triggers
│       ├── workflows.py          # Drift-triggered workflow records
│       ├── scheduler.py          # Policy evaluation for retraining workflows
│       ├── auth.py               # Token-based auth/authz helpers
│       ├── environment.py        # Dev/staging/production directory profiles
│       ├── logging.py            # Structured logging setup
│       ├── mcp_server.py         # MCP server entry point and tool dispatch
│       ├── ollama_client.py      # Ollama LLM client, recommendations, and analysis
│       ├── orchestration.py      # Training execution and resume coordination
│       ├── planning.py           # Recommendation-driven training plan creation
│       ├── pytorch_ops.py        # PyTorch MCP tool implementations
│       ├── task_queue.py         # Shared JSON-first task queue storage
│       ├── tensorflow_ops.py     # TensorFlow/Keras MCP tool implementations
│       ├── tool_catalog.py       # Transport-independent MCP tool registry
│       └── training.py           # Unified training pipeline
├── agents/
│   ├── __init__.py
│   └── training_agent.py        # Autonomous training agent
├── tests/
│   ├── architecture/            # Architecture contract tests
│   ├── integration/             # Cross-module integration tests
│   └── unit/                    # Focused unit tests
├── TASKS.json                   # Default object-backed training task queue
├── TASKS.md                     # Optional legacy markdown queue notes
├── AGENT.json                   # Object-backed agent runtime state
├── deploy/                      # Environment-level deployment notes
├── start.sh                     # Linux/macOS startup script
└── start.bat                    # Windows startup script
```

## MCP Tools

### PyTorch Tools

| Tool | Description |
|------|-------------|
| `pytorch_create_model` | Create a new PyTorch model (`resnet18`, `resnet50`, `mlp`, `custom`) |
| `pytorch_train_step` | Execute one training step with configurable batch size & LR |
| `pytorch_evaluate` | Evaluate model on dataset |
| `pytorch_save_checkpoint` | Save model checkpoint to disk |
| `pytorch_load_checkpoint` | Load model checkpoint from disk |

### TensorFlow Tools

| Tool | Description |
|------|-------------|
| `tensorflow_create_model` | Create a new TensorFlow/Keras model |
| `tensorflow_train_step` | Execute one training step |
| `tensorflow_evaluate` | Evaluate model on dataset |
| `tensorflow_save_checkpoint` | Save model checkpoint |
| `tensorflow_load_checkpoint` | Load model checkpoint |

### Training & Utility Tools

| Tool | Description |
|------|-------------|
| `get_training_status` | Get current training state, epochs completed, metrics count |
| `list_models` | List all registered models with backend info |
| `list_runs` | List persisted experiment and orchestration runs |
| `get_run` | Inspect a persisted run with dataset/code/artifact lineage |
| `compare_runs` | Compare historical runs by metrics and lineage |
| `register_model` | Register a checkpointed run as a model candidate |
| `list_registered_models` | List registry-managed model versions |
| `promote_model` | Promote an evaluated model to staging or production |
| `reject_model` | Reject a registry candidate with a reason |
| `predict_online` | Serve one prediction from a promoted registry model |
| `predict_batch` | Serve batch predictions from a promoted registry model |
| `get_model_observability` | Summarize latency, error rate, and class usage for served traffic |
| `get_drift_report` | Inspect drift reports derived from served traffic |
| `list_dataset_manifests` | List persisted dataset manifests |
| `get_dataset_manifest` | Inspect a dataset manifest by id and version |
| `get_dataset_validation_report` | Inspect a persisted dataset validation report |
| `list_training_tasks` | List queued agent tasks from the configured task queue |
| `create_training_task` | Create a queued training task in the configured task queue |
| `update_training_task` | Update task text or completion state in the configured task queue |
| `delete_training_task` | Delete a queued training task from the configured task queue |
| `get_system_health` | Report memory usage, cache sizes, and threshold status |
| `get_model_recommendation` | Get Ollama-powered model/architecture suggestions for a task |
| `get_training_analysis` | Ask Ollama to analyze a model's recorded training metrics and suggest next steps |

## Health Monitoring

Ghost now exposes resource health through both the training pipeline and MCP layer.

- `get_system_health` returns current system memory usage, GPU memory usage when available, cache sizes, and any active threshold violations.
- The training pipeline samples health before each epoch and reduces batch size when GPU or system memory crosses configured thresholds.
- Health samples are cached for `HEALTH_CHECK_INTERVAL` seconds so the interval setting now controls real sampling cadence instead of acting as documentation only.
- Cache directories are included in the health report so long-running sessions can inspect model and data footprint.
- MCP tool responses populate `structuredContent`, so Ghost returns object payloads to compatible clients instead of stringifying dictionaries into text.

## Production Stack Status

Ghost now covers the full local production-ML-stack loop inside the repository:

- versioned dataset manifests and validation reports
- searchable experiment records and artifact lineage
- evaluation-gated model registration and promotion
- registry-backed online and batch inference
- prediction observability, drift reporting, and alert derivation
- drift-triggered retraining workflow creation
- lightweight auth and environment separation primitives

The current implementation is Ghost-native and local-first. For team or internet-facing deployment, pair these abstractions with external infrastructure such as Postgres, S3/MinIO, a production scheduler, and a real HTTP serving deployment.

## Configuration Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default Ollama model | `llama3` |
| `OLLAMA_TIMEOUT` | Request timeout (seconds) | `60` |
| `TRAINING_BACKEND` | ML backend (`pytorch`/`tensorflow`/`auto`) | `auto` |
| `GPU_ENABLED` | Enable GPU acceleration | `true` |
| `CUDA_VISIBLE_DEVICES` | CUDA device selection | `0` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MODEL_CACHE_DIR` | Model checkpoint directory | `./models` |
| `DATA_CACHE_DIR` | Dataset cache directory | `./data` |
| `TASK_QUEUE_FILE` | Primary task queue file for the autonomous agent | `./TASKS.json` |
| `AGENT_STATE_FILE` | Object-backed agent state file | `./AGENT.json` |
| `DEFAULT_BATCH_SIZE` | Default training batch size | `32` |
| `DEFAULT_LEARNING_RATE` | Default learning rate | `0.001` |
| `DEFAULT_EPOCHS` | Default number of training epochs | `10` |
| `CHECKPOINT_INTERVAL` | Save checkpoint every N epochs | `5` |
| `MAX_ITERATIONS` | Max agent iterations before check-in | `100` |
| `ALLOW_SYNTHETIC_DATA` | Permit random demo batches when no dataset pipeline exists | `false` |
| `HEALTH_CHECK_INTERVAL` | Health check interval (seconds) | `30` |
| `GPU_MEMORY_THRESHOLD` | GPU memory usage alert threshold | `0.9` |
| `SYSTEM_MEMORY_THRESHOLD` | System memory usage alert threshold | `0.85` |

## Training Tasks

Use `TASKS.json` or the MCP task tools to queue training work for the autonomous agent:

```json
{
	"version": 1,
	"queue": [
		{"task_id": "cifar10-baseline", "text": "Train ResNet50 on CIFAR-10", "completed": false}
	]
}
```

The training agent watches `TASKS.json` by default and the MCP server can manage the same queue with `list_training_tasks`, `create_training_task`, `update_training_task`, and `delete_training_task`.

`TASKS.md` remains available only as an optional legacy input format. If you still want markdown, point the agent at a specific `.md` queue file explicitly.

When Ollama is available, Ghost now turns the task text into a lightweight training plan before execution. That plan influences the selected architecture plus the initial batch size, learning rate, and epochs while still falling back safely to the repository defaults when recommendations are missing or malformed.

Without a real dataset pipeline, these tasks will fail closed by default. Set `ALLOW_SYNTHETIC_DATA=true` only when you intentionally want demo-mode synthetic batches.

## Development Checks

Ghost's CI currently validates Python `3.10`, `3.11`, and `3.12` with linting, strict typing over `src/ghost`, security scans, and coverage enforcement.

```bash
ruff check src/ghost agents/
ruff format --check src/ghost agents/
mypy --ignore-missing-imports --strict src/ghost/
python -m pytest tests/ --cov=src/ghost --cov-report=term-missing --cov-fail-under=70
```

## License

MIT License — see LICENSE file for details.
