#!/bin/bash
# Push Ghost to GitHub - Run this from the Ghost directory

echo "Pushing Ghost to GitHub..."
echo ""

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Add all files
git add .

# Commit
git commit -m "Initial commit: Ghost - AI Model Context & Training Platform with PyTorch, TensorFlow, MCP, and Ollama"

