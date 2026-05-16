#!/bin/bash

set -e

echo "Check uv installation..."

if ! command -v uv &> /dev/null; then
    echo "uv is not installed."
    read -p "Install uv? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        echo "uv installed."
    else
        echo "Cannot continue without uv."
        exit 1
    fi
fi

echo "Syncing dependencies..."
uv sync
echo
echo "Setup complete."
echo
echo "Run programs with:"
echo "uv run python pipeline.py"
echo "Or activate env with source .venv/bin/activate and run python main.py"
echo 
echo "main_training.py for training a new model
test_GP.py for testing a trained model on the test set
pipeline.py for running the full pipeline on a single patient (useful for demo and debugging, not recommended for full dataset)
generic_dataset_preprocessing.py for preprocessing of custom dataset
custom_dataset_preprocessing.py for preprocessing of lymph dataset
"