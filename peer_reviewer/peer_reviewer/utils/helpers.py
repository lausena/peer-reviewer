"""
Helper utilities for the peer reviewer application.

This module provides common utility functions used across different components
of the peer review system.
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Union, Dict, Any, List
from datetime import datetime


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """
    Sanitize a filename for cross-platform compatibility.
    
    Args:
        filename: Original filename
        max_length: Maximum length for the filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace problematic characters
    safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    safe_filename = re.sub(r'\s+', '_', safe_filename)
    safe_filename = safe_filename.strip('._')
    
    # Ensure the filename isn't too long
    if len(safe_filename) > max_length:
        # Keep the extension if present
        parts = safe_filename.rsplit('.', 1)
        if len(parts) == 2 and len(parts[1]) <= 10:  # Reasonable extension length
            name_part = parts[0][:max_length - len(parts[1]) - 1]
            safe_filename = f"{name_part}.{parts[1]}"
        else:
            safe_filename = safe_filename[:max_length]
    
    return safe_filename or "unnamed_file"


def generate_file_hash(file_path: Path) -> str:
    """
    Generate MD5 hash for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def estimate_reading_time(text: str, wpm: int = 200) -> int:
    """
    Estimate reading time for text.
    
    Args:
        text: Text content
        wpm: Words per minute reading speed
        
    Returns:
        Estimated reading time in minutes
    """
    word_count = len(text.split())
    return max(1, round(word_count / wpm))


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with optional suffix.
    
    Args:
        text: Original text
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def clean_latex_text(text: str) -> str:
    """
    Clean text for LaTeX output by handling common issues.
    
    Args:
        text: Original text
        
    Returns:
        Cleaned text suitable for LaTeX
    """
    if not text:
        return ""
    
    # Handle unicode characters
    text = text.encode('ascii', errors='ignore').decode('ascii')
    
    # Fix common spacing issues
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Fix sentence spacing
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    return text.strip()


def safe_json_load(file_path: Path, default: Any = None) -> Any:
    """
    Safely load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default value if loading fails
        
    Returns:
        Loaded data or default value
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def safe_json_save(data: Any, file_path: Path) -> bool:
    """
    Safely save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        return True
    except Exception:
        return False


def validate_pdf_file(file_path: Path) -> Dict[str, Any]:
    """
    Validate that a file is a readable PDF.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "valid": False,
        "error": None,
        "file_size": 0,
        "exists": False
    }
    
    try:
        if not file_path.exists():
            result["error"] = "File does not exist"
            return result
        
        result["exists"] = True
        result["file_size"] = file_path.stat().st_size
        
        if result["file_size"] == 0:
            result["error"] = "File is empty"
            return result
        
        if not file_path.suffix.lower() == '.pdf':
            result["error"] = "File is not a PDF"
            return result
        
        # Try to read first few bytes to check PDF signature
        with open(file_path, 'rb') as f:
            header = f.read(8)
            if not header.startswith(b'%PDF-'):
                result["error"] = "File does not have valid PDF header"
                return result
        
        result["valid"] = True
        
    except Exception as e:
        result["error"] = f"Error validating file: {str(e)}"
    
    return result


def format_timestamp(timestamp: Union[datetime, str, None] = None) -> str:
    """
    Format timestamp for consistent display.
    
    Args:
        timestamp: Timestamp to format (defaults to current time)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    elif isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return timestamp  # Return original if parsing fails
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")