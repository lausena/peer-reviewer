#!/bin/bash

# Set memory optimization for M1 Mac
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Run peer-reviewer with memory optimizations
echo "Running peer-reviewer with memory optimizations..."
uv run peer-reviewer "$@"