"""Core peer review processing logic."""

import os
from pathlib import Path
from typing import List
from datetime import datetime

from .pdf_processor import extract_text_from_pdf, get_pdf_metadata
from .model_interface import generate_latex_review


def process_single_paper(paper_path: Path, reviews_dir: Path) -> None:
    """
    Process a single research paper and generate its peer review.
    
    Args:
        paper_path: Path to the PDF paper file
        reviews_dir: Directory to save the generated review
    """
    
    try:
        print(f"\n=== Processing Paper: {paper_path.name} ===")
        
        # Extract text from PDF
        print("1. Extracting text from PDF...")
        paper_content = extract_text_from_pdf(paper_path)
        
        if not paper_content.strip():
            print("Error: No text content extracted from PDF")
            return
        
        print(f"   Extracted {len(paper_content)} characters")
        
        # Extract metadata for context
        print("2. Extracting PDF metadata...")
        metadata = get_pdf_metadata(paper_path)
        print(f"   Title: {metadata.get('title', 'N/A')}")
        print(f"   Author: {metadata.get('author', 'N/A')}")
        print(f"   Pages: {metadata.get('page_count', 'N/A')}")
        
        # Generate LaTeX review using optimized gpt-oss-20b
        print("3. Generating LaTeX peer review (optimized for speed)...")
        latex_review = generate_latex_review(paper_content)
        
        if not latex_review.strip():
            print("Error: No LaTeX review generated")
            return
        
        # Create output filename
        paper_stem = paper_path.stem  # filename without extension
        output_filename = f"{paper_stem}_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
        output_path = reviews_dir / output_filename
        
        # Save the LaTeX review
        print("4. Saving LaTeX review...")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_review)
        
        print(f"✅ Review generated successfully!")
        print(f"   Output: {output_path}")
        print(f"   Size: {len(latex_review)} characters")
        
        # Optional: Print first few lines for verification
        lines = latex_review.split('\n')[:10]
        print(f"\n   Preview (first 10 lines):")
        for i, line in enumerate(lines, 1):
            print(f"   {i:2d}: {line}")
        if len(latex_review.split('\n')) > 10:
            print(f"   ... ({len(latex_review.split('\n')) - 10} more lines)")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Error processing paper {paper_path.name}: {e}")


def process_all_papers(papers_dir: Path, reviews_dir: Path) -> None:
    """
    Process all PDF papers in the papers directory.
    
    Args:
        papers_dir: Directory containing PDF papers
        reviews_dir: Directory to save generated reviews
    """
    
    # Find all PDF files in papers directory
    pdf_files = list(papers_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {papers_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    # Process each paper
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(pdf_files)}: {pdf_file.name}")
        print(f"{'='*60}")
        
        process_single_paper(pdf_file, reviews_dir)
    
    print(f"\n🎉 Completed processing {len(pdf_files)} paper(s)!")
    print(f"Reviews saved to: {reviews_dir}")


def get_paper_info(paper_path: Path) -> dict:
    """
    Get basic information about a paper file.
    
    Args:
        paper_path: Path to the PDF paper
        
    Returns:
        Dictionary with paper information
    """
    
    try:
        # Get file stats
        file_stats = paper_path.stat()
        file_size_mb = file_stats.st_size / (1024 * 1024)
        
        # Get PDF metadata
        metadata = get_pdf_metadata(paper_path)
        
        return {
            "filename": paper_path.name,
            "filepath": str(paper_path),
            "size_mb": round(file_size_mb, 2),
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "pages": metadata.get("page_count", 0),
            "creation_date": metadata.get("creation_date", ""),
        }
        
    except Exception as e:
        return {
            "filename": paper_path.name,
            "filepath": str(paper_path),
            "error": str(e)
        }