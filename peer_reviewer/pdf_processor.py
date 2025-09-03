"""PDF text extraction functionality using PyMuPDF."""

import fitz  # PyMuPDF
from pathlib import Path


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text content from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content as a string
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If PDF extraction fails
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        # Open the PDF document
        doc = fitz.open(str(pdf_path))
        
        # Extract text from all pages
        text_content = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Add page separator for better structure
            if page_num > 0:
                text_content.append(f"\n\n--- Page {page_num + 1} ---\n\n")
            
            text_content.append(text)
        
        # Close the document
        doc.close()
        
        # Join all text content
        full_text = "".join(text_content)
        
        # Basic text cleaning
        full_text = full_text.strip()
        
        # Remove excessive whitespace while preserving paragraph breaks
        lines = full_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line or (cleaned_lines and cleaned_lines[-1]):  # Keep empty lines that separate paragraphs
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
        
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF {pdf_path}: {str(e)}")


def get_pdf_metadata(pdf_path: Path) -> dict:
    """
    Extract metadata from PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing PDF metadata
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        doc = fitz.open(str(pdf_path))
        metadata = doc.metadata
        doc.close()
        
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
            "page_count": len(doc) if doc else 0
        }
        
    except Exception as e:
        return {"error": f"Failed to extract metadata: {str(e)}"}