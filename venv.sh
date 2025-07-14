#!/bin/bash

# Set the path to your venv directory
VENV_DIR="./.venv"
REQ_FILE="./requirements.txt"

# Step 1: Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "🛠️  No virtual environment found. Creating one at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment. Make sure Python 3 is installed."
        exit 1
    fi
    echo "✅ Virtual environment created."
fi

# Step 2: Activate the virtual environment
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated."

# Step 3: Install requirements if file exists
if [ -f "$REQ_FILE" ]; then
    echo "📦 Installing packages from $REQ_FILE..."
    pip install --upgrade pip
    pip install -r "$REQ_FILE"
    echo "✅ Requirements installed."
else
    echo "⚠️  No requirements.txt found. Skipping package installation."
fi

# Step 4: Drop into a new shell with venv active
exec $SHELL