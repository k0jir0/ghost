#!/bin/bash
# Ghost Platform Startup Script for Linux/macOS

set -e

echo "Starting Ghost Platform..."
echo ""

# Set Python path
export PYTHONPATH="${PWD}/src"
export GHOST_LOG_FILE="ghost.log"

# Parse command
case "${1:-server}" in
    server)
        echo "Starting MCP Server..."
        python -m ghost.mcp_server
        ;;
    agent)
        echo "Starting Training Agent..."
        python -m agents.training_agent
        ;;
    daemon)
        echo "Starting Agent in daemon mode..."
        nohup python -m agents.training_agent > ghost.log 2>&1 &
        echo "Agent running in background (PID: $!)"
        ;;
    cli)
        echo "Starting Ghost Operator CLI..."
        python -m ghost.cli
        ;;
    install)
        echo "Installing Ghost..."
        pip install -e .
        echo "Ghost installed successfully!"
        ;;
    *)
        echo "Usage: ./start.sh [server|agent|daemon|cli|install]"
        echo ""
        echo "Commands:"
        echo "  server   - Start MCP server (default)"
        echo "  agent    - Start training agent"
        echo "  daemon   - Start agent in background"
        echo "  cli      - Start numbered Ghost operator CLI"
        echo "  install  - Install Ghost package"
        exit 1
        ;;
esac
