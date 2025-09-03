# Peer Reviewer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An AI-powered peer review system that generates comprehensive LaTeX reviews for research papers using OpenAI's gpt-oss-20b model. The system processes PDF files and creates professional, publication-ready review documents.

## Features

- **Automated PDF Processing**: Extract and analyze content from research papers
- **AI-Powered Reviews**: Generate comprehensive peer reviews using gpt-oss-20b
- **LaTeX Output**: Professional, publication-ready review documents
- **Flexible CLI**: Rich command-line interface with multiple configuration options
- **Memory Optimization**: Built-in memory management for resource-constrained environments
- **Batch Processing**: Review multiple papers or individual documents
- **Configurable Parameters**: Customizable review types, reasoning levels, and model settings

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd peer-reviewer
```

2. Install dependencies using uv:
```bash
uv sync
```

## Usage

### Quick Start

```bash
# Review all papers in the papers/ directory
uv run peer-reviewer

# Review a specific paper
uv run peer-reviewer papers/my-paper.pdf

# Quick demo mode (fast template generation)
./demo_review.sh

# Memory-optimized run (for M1 Macs)
./run_memory_optimized.sh
```

### Advanced Usage

The CLI provides extensive configuration options:

```bash
# Use different model settings
uv run peer-reviewer --temperature 0.5 --reasoning medium --type brief

# Custom directories
uv run peer-reviewer --papers-dir /path/to/papers --reviews-dir /path/to/output

# Force overwrite existing reviews
uv run peer-reviewer --force

# Verbose logging
uv run peer-reviewer --verbose
```

### Available Commands

#### Review Command
```bash
uv run peer-reviewer review [OPTIONS] [PAPER_PATH]
```

**Options:**
- `--papers-dir, -p`: Directory containing papers to review (default: papers)
- `--reviews-dir, -r`: Directory to save generated reviews (default: reviews)
- `--model, -m`: Model name to use (default: microsoft/gpt-oss-20b)
- `--reasoning, -l`: Reasoning level (high, medium, low)
- `--type, -t`: Review type (comprehensive, brief, technical)
- `--temperature`: Temperature for text generation (0.0-1.0)
- `--max-tokens`: Maximum tokens for generated review
- `--force, -f`: Overwrite existing review files
- `--verbose, -v`: Enable verbose logging

#### Info Command
```bash
uv run peer-reviewer info [OPTIONS]
```

Shows system information and available papers.

#### Config Command
```bash
uv run peer-reviewer config [OPTIONS]
```

Manage application configuration:
- `--show, -s`: Show current configuration
- `--reset`: Reset to default configuration
- `--model`: Set default model name
- `--reasoning`: Set default reasoning level
- `--temperature`: Set default temperature

## Project Structure

```
peer-reviewer/
├── peer_reviewer/           # Main application package
│   ├── __init__.py
│   ├── __main__.py         # Main entry point
│   ├── core.py             # Core processing logic
│   ├── model_interface.py  # AI model interface
│   ├── pdf_processor.py    # PDF processing utilities
│   └── peer_reviewer/      # Extended CLI and utilities
│       ├── cli.py          # Rich CLI interface
│       ├── core/           # Core components
│       │   ├── reviewer.py
│       │   ├── model_interface.py
│       │   ├── latex_generator.py
│       │   └── pdf_processor.py
│       ├── utils/          # Utility modules
│       │   ├── config.py
│       │   ├── logging.py
│       │   ├── helpers.py
│       │   └── exceptions.py
│       └── templates/      # LaTeX templates
├── papers/                 # Input directory for PDF papers
├── reviews/               # Output directory for generated reviews
├── tests/                 # Test files
├── demo_review.sh         # Quick demo script
├── run_memory_optimized.sh # Memory-optimized execution
└── pyproject.toml         # Project configuration
```

## Configuration

The application supports configuration through:

1. **Command-line arguments**: Override default settings per run
2. **Configuration file**: Persistent settings via the `config` command
3. **Environment variables**: For system-specific optimizations

### Memory Optimization

For resource-constrained environments (like M1 Macs), use:

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
./run_memory_optimized.sh
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --dev

# Run tests
python -m pytest

# Code formatting
black .
isort .

# Type checking
mypy peer_reviewer/

# Linting
flake8 peer_reviewer/
```

### Running Tests

```bash
# Basic functionality test
python test_basic.py

# Demo test
python test_demo.py

# Full test suite (if available)
pytest
```

## Output

The system generates LaTeX review files in the `reviews/` directory with the following structure:

- Professional academic formatting
- Comprehensive review sections (Abstract, Introduction, Methods, Results, Discussion, Conclusions)
- Author contributions and competing interests
- Properly formatted bibliography
- Publication-ready styling

Generated files follow the naming convention: `<paper_name>_review.tex`

## Dependencies

### Core Dependencies
- **torch**: Deep learning framework
- **transformers**: Hugging Face transformers library
- **accelerate**: Hardware acceleration
- **pymupdf**: PDF processing
- **tokenizers**: Text tokenization

### CLI Dependencies
- **typer**: Modern CLI framework
- **rich**: Rich terminal output
- **loguru**: Advanced logging

### Development Dependencies
- **pytest**: Testing framework
- **black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **flake8**: Linting

## Requirements

- Python 3.10+
- Sufficient RAM for model loading (varies by model size)
- GPU optional but recommended for faster processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run the test suite
6. Submit a pull request

## Troubleshooting

### Common Issues

**Memory Issues on M1 Macs:**
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

**Model Loading Errors:**
Ensure you have sufficient RAM and the model is properly downloaded.

**PDF Processing Errors:**
Verify that PDF files are not corrupted and are readable.

**Permission Errors:**
Ensure write permissions for the `reviews/` directory.

For more issues, check the logs with `--verbose` flag or consult the documentation.