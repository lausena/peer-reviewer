#!/usr/bin/env python3
"""
Simplified peer review application using gpt-oss-20b for LaTeX generation.

Usage:
    uv run peer-reviewer                    # Review all papers in papers/ directory
    uv run peer-reviewer papers/paper.pdf   # Review specific paper
"""

import sys
import os
from pathlib import Path
from typing import Optional

from .pdf_processor import extract_text_from_pdf
from .model_interface import generate_latex_review
from .core import process_single_paper, process_all_papers


def main():
    """Main entry point for the peer review application."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    papers_dir = project_root / "papers"
    reviews_dir = project_root / "reviews"
    
    # Ensure reviews directory exists
    reviews_dir.mkdir(exist_ok=True)
    
    # Parse command line arguments
    if len(sys.argv) == 1:
        # No arguments: process all papers in papers/ directory
        print("Processing all papers in papers/ directory (optimized)...")
        process_all_papers(papers_dir, reviews_dir)
    elif len(sys.argv) == 2:
        # One argument: process specific paper
        paper_path = Path(sys.argv[1])
        if not paper_path.exists():
            print(f"Error: Paper file '{paper_path}' not found.")
            sys.exit(1)
        
        print(f"Processing paper: {paper_path} (optimized)")
        process_single_paper(paper_path, reviews_dir)
    else:
        print("Usage:")
        print("  uv run peer-reviewer                    # Review all papers")
        print("  uv run peer-reviewer papers/paper.pdf   # Review specific paper")
        sys.exit(1)


if __name__ == "__main__":
    main()