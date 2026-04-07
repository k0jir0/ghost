# Ghost — AI Model Context & Training Platform

Ghost is an intelligent ML training and inference platform that combines **PyTorch** and **TensorFlow** backends with the **Model Context Protocol (MCP)** for context-aware AI interactions and **Ollama** for local LLM-powered assistance. The current codebase is organized around explicit planning, dataset resolution, orchestration, and tool-catalog layers so the autonomous agent and MCP server share the same training primitives.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      GHOST PLATFORM                         │
├─────────────────────────────────────────────────────────────┤
│  Agent + MCP Entry Points                                   │
│  ├── TrainingAgent (TASKS.md automation)                    │
│  └── GhostMCPServer (tool transport + dispatch)             │
├─────────────────────────────────────────────────────────────┤
│  Application Services                                       │
│  ├── planning.py        (recommendation-driven plans)       │
│  ├── datasets.py        (dataset resolution + demo gating)  │
│  ├── orchestration.py   (run execution + resume flow)       │
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
- **Autonomous Training Agent** — Watches `TASKS.md` and executes training tasks automatically
- **Recommendation-Driven Planning** — The agent uses Ollama recommendations to select architecture and training hyperparameters before execution
- **Shared Service Layer** — Planning, dataset resolution, orchestration, and tool-catalog modules keep the agent and MCP server aligned
- **Explicit Demo Mode** — Synthetic training and evaluation data is disabled by default and must be opted into for scaffold/demo runs
- **Health Monitoring** — GPU, memory, and cache monitoring with configurable thresholds
- **Graceful Degradation** — Memory pressure automatically reduces training batch size before the run escalates
- **Type-Safe Config** — Pydantic-based `GhostConfig` with `.env` file support

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
│       ├── datasets.py           # Dataset resolution and demo-mode policy
│       ├── health_monitor.py     # Resource monitoring and adaptive health checks
│       ├── logging.py            # Structured logging setup
│       ├── mcp_server.py         # MCP server entry point and tool dispatch
│       ├── ollama_client.py      # Ollama LLM client, recommendations, and analysis
│       ├── orchestration.py      # Training execution and resume coordination
│       ├── planning.py           # Recommendation-driven training plan creation
│       ├── pytorch_ops.py        # PyTorch MCP tool implementations
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
├── TASKS.md                     # Training task queue (agent reads this)
├── AGENT.md                     # Agent memory & state
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
| `get_system_health` | Report memory usage, cache sizes, and threshold status |
| `get_model_recommendation` | Get Ollama-powered model/architecture suggestions for a task |
| `get_training_analysis` | Ask Ollama to analyze a model's recorded training metrics and suggest next steps |

## Health Monitoring

Ghost now exposes resource health through both the training pipeline and MCP layer.

- `get_system_health` returns current system memory usage, GPU memory usage when available, cache sizes, and any active threshold violations.
- The training pipeline samples health before each epoch and reduces batch size when GPU or system memory crosses configured thresholds.
- Health samples are cached for `HEALTH_CHECK_INTERVAL` seconds so the interval setting now controls real sampling cadence instead of acting as documentation only.
- Cache directories are included in the health report so long-running sessions can inspect model and data footprint.

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

Edit `TASKS.md` to queue training tasks for the autonomous agent:

```markdown
## Queue

- [ ] Train ResNet50 on CIFAR-10
- [ ] Fine-tune BERT for sentiment analysis
- [ ] Experiment with learning rate schedules
```

The training agent will pick them up automatically on the next iteration.

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
