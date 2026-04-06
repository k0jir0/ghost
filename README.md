# Ghost — AI Model Context & Training Platform

Ghost is an intelligent ML training and inference platform that combines **PyTorch** and **TensorFlow** backends with the **Model Context Protocol (MCP)** for context-aware AI interactions and **Ollama** for local LLM-powered assistance.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      GHOST PLATFORM                          │
├─────────────────────────────────────────────────────────────┤
│  MCP Server Layer (GhostMCPServer)                           │
│  ├── Context Manager (model state, training history)         │
│  ├── Tool Registry (training ops, inference, metrics)        │
│  └── Health Monitor (GPU, memory, model cache)               │
├─────────────────────────────────────────────────────────────┤
│  ML Backends                                                 │
│  ├── PyTorch Integration  (pytorch_ops.py)                   │
│  ├── TensorFlow/Keras Integration  (tensorflow_ops.py)       │
│  └── Unified Training Pipeline  (training.py)               │
├─────────────────────────────────────────────────────────────┤
│  Ollama Layer (ollama_client.py)                             │
│  ├── Local LLM inference                                     │
│  ├── Training assistance AI                                  │
│  └── Model recommendation engine                             │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Dual Framework Support** — Seamless PyTorch and TensorFlow integration with a unified API
- **MCP Protocol** — Standardized tool interface for AI model interactions via `GhostMCPServer`
- **Local LLM** — Ollama integration for private, offline inference and model recommendations
- **Autonomous Training Agent** — Watches `TASKS.md` and executes training tasks automatically
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

# Install dependencies
pip install torch tensorflow mcp ollama pydantic pydantic-settings structlog python-dotenv

# (Optional) Pull Ollama models
ollama pull llama3
ollama pull mistral
```

### Configuration

```bash
# Create your .env file
cp .env.example .env   # or create manually

# Key settings:
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_MODEL=llama3
# TRAINING_BACKEND=auto          # pytorch | tensorflow | auto
# GPU_ENABLED=true
# LOG_LEVEL=INFO
# MODEL_CACHE_DIR=./models
# DATA_CACHE_DIR=./data
```

### Running

```bash
# Start the MCP server
python -m ghost.mcp_server

# Or use the startup scripts
./start.sh       # Linux/macOS
start.bat        # Windows

# Run the autonomous training agent
python -m ghost.agent
```

## Project Structure

```
ghost/
├── src/
│   └── ghost/
│       ├── __init__.py           # Package init — GhostConfig, ModelContext, TrainingPipeline
│       ├── mcp_server.py         # MCP server — GhostMCPServer with all registered tools
│       ├── pytorch_ops.py        # PyTorch MCP tool implementations
│       ├── tensorflow_ops.py     # TensorFlow/Keras MCP tool implementations
│       ├── ollama_client.py      # Ollama LLM client & recommendation engine
│       ├── training.py           # Unified training pipeline
│       ├── context.py            # Model context & state management
│       ├── config.py             # Pydantic-based configuration (GhostConfig)
│       ├── health_monitor.py     # Resource monitoring and adaptive health checks
│       └── logging.py            # Structured logging setup
├── agents/
│   ├── __init__.py
│   └── training_agent.py        # Autonomous training agent
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

## Health Monitoring

Ghost now exposes resource health through both the training pipeline and MCP layer.

- `get_system_health` returns current system memory usage, GPU memory usage when available, cache sizes, and any active threshold violations.
- The training pipeline samples health before each epoch and reduces batch size when GPU or system memory crosses configured thresholds.
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
| `DAILY_TOKEN_BUDGET` | Daily AI token budget (USD) | `10.0` |
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

## License

MIT License — see LICENSE file for details.
