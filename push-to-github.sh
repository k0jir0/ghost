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

# Create GitHub repo and push (run these commands manually if needed):
echo ""
echo "To create GitHub repo and push, run:"
echo "1. Create repo on GitHub.com"
echo "2. git remote add origin https://github.com/YOUR_USERNAME/ghost.git"
echo "3. git branch -M main"
echo "4. git push -u origin main"
