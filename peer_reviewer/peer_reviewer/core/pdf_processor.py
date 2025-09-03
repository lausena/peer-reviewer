"""
PDF processing module for extracting text from research papers.

This module provides robust PDF text extraction capabilities using both PyPDF2 and pymupdf
as fallback options to handle various PDF formats and encoding issues.
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

import fitz  # pymupdf
import PyPDF2
from loguru import logger


@dataclass
class DocumentMetadata:
    """Metadata extracted from PDF documents."""
    title: Optional[str] = None
    authors: Optional[str] = None
    abstract: Optional[str] = None
    keywords: Optional[str] = None
    page_count: int = 0
    file_size: int = 0
    creation_date: Optional[str] = None


class PDFProcessor:
    """
    Advanced PDF text extraction with support for academic papers.
    
    This class handles text extraction from PDF files with special attention to
    academic paper structure including title, authors, abstract, and main content.
    """
    
    def __init__(self, min_text_length: int = 100):
        """
        Initialize PDF processor.
        
        Args:
            min_text_length: Minimum text length to consider extraction successful
        """
        self.min_text_length = min_text_length
        logger.info("PDF processor initialized")
    
    def extract_text(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text and metadata from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text, metadata, and processing info
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF extraction fails
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        # Try pymupdf first (better for academic papers)
        try:
            result = self._extract_with_pymupdf(pdf_path)
            if len(result["text"]) >= self.min_text_length:
                logger.success(f"Successfully extracted {len(result['text'])} characters using pymupdf")
                return result
        except Exception as e:
            logger.warning(f"pymupdf extraction failed: {e}")
        
        # Fallback to PyPDF2
        try:
            result = self._extract_with_pypdf2(pdf_path)
            if len(result["text"]) >= self.min_text_length:
                logger.success(f"Successfully extracted {len(result['text'])} characters using PyPDF2")
                return result
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            
        raise ValueError(f"Failed to extract sufficient text from {pdf_path.name}")
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text using pymupdf (fitz)."""
        doc = fitz.open(str(pdf_path))
        
        # Extract metadata
        metadata = self._extract_metadata_pymupdf(doc)
        
        # Extract text from all pages
        full_text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            full_text += text + "\n\n"
        
        doc.close()
        
        # Clean and structure the text
        cleaned_text = self._clean_text(full_text)
        structured_content = self._structure_academic_content(cleaned_text)
        
        return {
            "text": cleaned_text,
            "structured_content": structured_content,
            "metadata": metadata,
            "extraction_method": "pymupdf",
            "success": True
        }
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text using PyPDF2 as fallback."""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract metadata
            metadata = self._extract_metadata_pypdf2(pdf_reader, pdf_path)
            
            # Extract text from all pages
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    full_text += text + "\n\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
        
        # Clean and structure the text
        cleaned_text = self._clean_text(full_text)
        structured_content = self._structure_academic_content(cleaned_text)
        
        return {
            "text": cleaned_text,
            "structured_content": structured_content,
            "metadata": metadata,
            "extraction_method": "PyPDF2",
            "success": True
        }
    
    def _extract_metadata_pymupdf(self, doc: fitz.Document) -> DocumentMetadata:
        """Extract metadata using pymupdf."""
        try:
            metadata_dict = doc.metadata
            return DocumentMetadata(
                title=metadata_dict.get("title"),
                authors=metadata_dict.get("author"),
                keywords=metadata_dict.get("keywords"),
                page_count=doc.page_count,
                creation_date=metadata_dict.get("creationDate")
            )
        except Exception as e:
            logger.warning(f"Failed to extract metadata with pymupdf: {e}")
            return DocumentMetadata(page_count=doc.page_count)
    
    def _extract_metadata_pypdf2(self, pdf_reader: PyPDF2.PdfReader, pdf_path: Path) -> DocumentMetadata:
        """Extract metadata using PyPDF2."""
        try:
            metadata = DocumentMetadata(
                page_count=len(pdf_reader.pages),
                file_size=pdf_path.stat().st_size
            )
            
            if pdf_reader.metadata:
                metadata.title = pdf_reader.metadata.get("/Title")
                metadata.authors = pdf_reader.metadata.get("/Author")
                metadata.creation_date = pdf_reader.metadata.get("/CreationDate")
                
            return metadata
        except Exception as e:
            logger.warning(f"Failed to extract metadata with PyPDF2: {e}")
            return DocumentMetadata(page_count=len(pdf_reader.pages))
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text, flags=re.IGNORECASE)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between joined words
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after sentences
        
        # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _structure_academic_content(self, text: str) -> Dict[str, str]:
        """
        Extract structured content from academic papers.
        
        Attempts to identify common sections in academic papers.
        """
        structured = {
            "title": "",
            "abstract": "",
            "introduction": "",
            "main_content": "",
            "conclusion": "",
            "references": ""
        }
        
        if not text:
            return structured
        
        # Try to extract title (usually first significant text)
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.lower().startswith(('page', 'vol', 'doi')):
                structured["title"] = line
                break
        
        # Try to extract abstract
        abstract_match = re.search(
            r'(?:^|\n)\s*(?:ABSTRACT|Abstract)\s*[:\-]?\s*\n(.+?)(?=\n\s*(?:KEYWORDS|Keywords|INTRODUCTION|Introduction|1\.|\n\n))',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if abstract_match:
            structured["abstract"] = abstract_match.group(1).strip()
        
        # Try to extract introduction
        intro_match = re.search(
            r'(?:^|\n)\s*(?:1\.|INTRODUCTION|Introduction)\s*\n(.+?)(?=\n\s*(?:2\.|RELATED|Related|METHOD|Method))',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if intro_match:
            structured["introduction"] = intro_match.group(1).strip()
        
        # Try to extract references
        ref_match = re.search(
            r'(?:^|\n)\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)\s*\n(.+?)$',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if ref_match:
            structured["references"] = ref_match.group(1).strip()
        
        # Main content is everything else
        structured["main_content"] = text
        
        return structured
    
    def get_document_stats(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Get basic statistics about the PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with document statistics
        """
        try:
            extraction_result = self.extract_text(pdf_path)
            text = extraction_result["text"]
            metadata = extraction_result["metadata"]
            
            word_count = len(text.split())
            char_count = len(text)
            
            stats = {
                "file_name": pdf_path.name,
                "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
                "page_count": metadata.page_count,
                "word_count": word_count,
                "character_count": char_count,
                "estimated_reading_time_minutes": round(word_count / 200),  # ~200 WPM average
                "extraction_method": extraction_result["extraction_method"]
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get document stats: {e}")
            return {"error": str(e)}