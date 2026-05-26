# Ghost — AI Model Context & Training Platform

Ghost is an intelligent ML training and inference platform that combines **PyTorch** and **TensorFlow** backends with the **Model Context Protocol (MCP)** for context-aware AI interactions and **Ollama** for local LLM-powered assistance. The current codebase is organized around explicit planning, dataset resolution, orchestration, and tool-catalog layers so the autonomous agent and MCP server share the same training primitives.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      GHOST PLATFORM                         │
├─────────────────────────────────────────────────────────────┤
│  Agent + MCP Entry Points                                   │
│  ├── TrainingAgent (TASKS.json queue, AGENT.json state)     │
│  └── GhostMCPServer (tool transport + dispatch)             │
├─────────────────────────────────────────────────────────────┤
│  Application Services                                       │
│  ├── planning.py        (recommendation-driven plans)       │
│  ├── datasets.py        (dataset resolution + demo gating)  │
│  ├── data_loading.py    (real dataset loading + batching)   │
│  ├── orchestration.py   (run execution + resume flow)       │
│  ├── task_queue.py      (shared JSON/markdown task store)   │
│  ├── tool_catalog.py    (transport-independent tool specs)  │
│  └── health_monitor.py  (resource-aware training signals)   │
├─────────────────────────────────────────────────────────────┤
│  Runtime + Backends                                         │
│  ├── context.py / training.py                               │
│  ├── pytorch_ops.py                                          │
│  └── tensorflow_ops.py                                       │
├─────────────────────────────────────────────────────────────┤
│  Ollama Layer (ollama_client.py)                            │
│  ├── Local LLM inference                                    │
│  ├── Model recommendation engine                            │
│  └── Training analysis from recorded metrics                │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Dual Framework Support** — Seamless PyTorch and TensorFlow integration with a unified API
- **MCP Protocol** — Standardized tool interface for AI model interactions via `GhostMCPServer`
- **Local LLM** — Ollama integration for private, offline inference and model recommendations
- **Autonomous Training Agent** — Watches `TASKS.json` by default, exposes queue CRUD through MCP tools, and persists machine state in `AGENT.json`
- **Recommendation-Driven Planning** — The agent uses Ollama recommendations to select architecture and training hyperparameters before execution
- **Real Dataset Runtime** — CIFAR-10, MNIST, and IMDB-review datasets now flow through a shared cached loader and batch provider instead of synthetic-only scaffolding
- **Shared Service Layer** — Planning, dataset resolution, orchestration, and tool-catalog modules keep the agent and MCP server aligned
- **Explicit Demo Mode** — Synthetic training and evaluation data is disabled by default and must be opted into for scaffold/demo runs
- **Health Monitoring** — GPU, memory, and cache monitoring with configurable thresholds
- **Graceful Degradation** — Memory pressure automatically reduces training batch size before the run escalates
- **Type-Safe Config** — Pydantic-based `GhostConfig` with `.env` file support

## Recent Progress

- Real dataset wiring now resolves dataset specs into backend-ready input shapes and feeds repeatable batches to both PyTorch and TensorFlow backends through `data_loading.py`.
- The autonomous agent now stores machine-managed state in `AGENT.json`, defaults its queue to `TASKS.json`, and shares queue semantics with the MCP layer through `task_queue.py`.
- MCP tool responses now use `structuredContent`, preserving object payloads instead of stringifying dictionaries into text.
- The current regression baseline is `182 passed` across architecture, integration, and unit tests.

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
│       ├── health_monitor.py     # Resource monitoring and adaptive health checks
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
