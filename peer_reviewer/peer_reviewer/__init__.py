"""
Peer Reviewer: AI-powered peer review system for research papers.

This package provides automated peer review capabilities using OpenAI's gpt-oss-20b model
to analyze research papers and generate comprehensive LaTeX-formatted reviews.
"""

__version__ = "1.0.0"
__author__ = "Peer Review System"
__email__ = "peer-reviewer@example.com"

from .core.reviewer import PeerReviewer
from .core.pdf_processor import PDFProcessor
from .core.model_interface import ModelInterface
from .core.latex_generator import LaTeXGenerator

__all__ = [
    "PeerReviewer",
    "PDFProcessor", 
    "ModelInterface",
    "LaTeXGenerator",
]