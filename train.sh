#!/bin/bash
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    python train.py
else
    echo "Virtual environment not found. Please create one with './install.sh' and install dependencies."
    exit 1
fi
    