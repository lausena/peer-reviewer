"""
Advanced logging configuration for the peer reviewer application.

This module provides structured logging with proper formatting, file output,
and performance monitoring capabilities.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import traceback

from loguru import logger


class StructuredLogger:
    """
    Structured logger with JSON output and performance tracking.
    
    This class provides enhanced logging capabilities with structured output,
    performance timing, and proper error context tracking.
    """
    
    def __init__(
        self,
        name: str = "peer-reviewer",
        log_dir: Optional[Path] = None,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        enable_json: bool = False
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name/component
            log_dir: Directory for log files
            console_level: Console logging level
            file_level: File logging level
            enable_json: Enable JSON structured logging
        """
        self.name = name
        self.log_dir = log_dir or Path("logs")
        self.console_level = console_level.upper()
        self.file_level = file_level.upper()
        self.enable_json = enable_json
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Configure loguru logger with proper handlers."""
        # Remove default handler
        logger.remove()
        
        # Console handler with colored output
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        logger.add(
            sys.stderr,
            format=console_format,
            level=self.console_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # File handler for general logs
        log_file = self.log_dir / f"{self.name}.log"
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        
        logger.add(
            str(log_file),
            format=file_format,
            level=self.file_level,
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            backtrace=True,
            diagnose=True
        )
        
        # Error-specific file handler
        error_file = self.log_dir / f"{self.name}_errors.log"
        logger.add(
            str(error_file),
            format=file_format,
            level="ERROR",
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=True
        )
        
        # JSON structured log handler (optional)
        if self.enable_json:
            json_file = self.log_dir / f"{self.name}_structured.jsonl"
            logger.add(
                str(json_file),
                format=self._json_formatter,
                level=self.file_level,
                rotation="10 MB",
                retention="7 days",
                compression="gz"
            )
    
    def _json_formatter(self, record) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "module": record["module"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
        }
        
        # Add extra fields if present
        if "extra" in record:
            log_entry.update(record["extra"])
        
        # Add exception info if present
        if record["exception"]:
            log_entry["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback
            }
        
        return json.dumps(log_entry)
    
    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        """
        Log performance metrics.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            **kwargs: Additional performance metrics
        """
        metrics = {
            "operation": operation,
            "duration_seconds": round(duration, 3),
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        logger.bind(**metrics).info(f"Performance: {operation} completed in {duration:.3f}s")
    
    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """
        Log model information and configuration.
        
        Args:
            model_info: Dictionary containing model information
        """
        logger.bind(**model_info).info("Model information logged")
    
    def log_processing_stats(self, stats: Dict[str, Any]) -> None:
        """
        Log document processing statistics.
        
        Args:
            stats: Dictionary containing processing statistics
        """
        logger.bind(**stats).info("Document processing statistics")
    
    def log_error_with_context(
        self,
        error: Exception,
        context: Dict[str, Any],
        operation: str = ""
    ) -> None:
        """
        Log error with detailed context information.
        
        Args:
            error: Exception that occurred
            context: Context information
            operation: Operation that was being performed
        """
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "operation": operation,
            "context": context,
            "traceback": traceback.format_exc()
        }
        
        logger.bind(**error_context).error(f"Error in {operation}: {error}")


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation: str, logger_instance: Optional[StructuredLogger] = None):
        """
        Initialize performance timer.
        
        Args:
            operation: Name of the operation being timed
            logger_instance: Logger instance to use for logging
        """
        self.operation = operation
        self.logger = logger_instance
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        logger.info(f"Starting {self.operation}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log results."""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            logger.success(f"Completed {self.operation} in {self.duration:.3f}s")
            if self.logger:
                self.logger.log_performance(self.operation, self.duration)
        else:
            logger.error(f"Failed {self.operation} after {self.duration:.3f}s")
            if self.logger:
                self.logger.log_error_with_context(
                    exc_val,
                    {"duration": self.duration},
                    self.operation
                )


def setup_application_logging(
    verbose: bool = False,
    log_dir: Optional[Path] = None,
    enable_json: bool = False
) -> StructuredLogger:
    """
    Setup application-wide logging configuration.
    
    Args:
        verbose: Enable verbose logging
        log_dir: Directory for log files
        enable_json: Enable JSON structured logging
        
    Returns:
        Configured StructuredLogger instance
    """
    console_level = "DEBUG" if verbose else "INFO"
    
    structured_logger = StructuredLogger(
        name="peer-reviewer",
        log_dir=log_dir,
        console_level=console_level,
        file_level="DEBUG",
        enable_json=enable_json
    )
    
    # Log application startup
    logger.info("Peer reviewer application logging initialized")
    logger.info(f"Console level: {console_level}")
    logger.info(f"Log directory: {structured_logger.log_dir}")
    logger.info(f"JSON logging: {enable_json}")
    
    return structured_logger


def log_system_info() -> None:
    """Log system information for debugging."""
    import platform
    import psutil
    import torch
    
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    }
    
    if torch.cuda.is_available():
        system_info["cuda_version"] = torch.version.cuda
        system_info["cuda_device_count"] = torch.cuda.device_count()
        system_info["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
    
    logger.bind(**system_info).info("System information logged")


# Global logger instance
app_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """Get the global application logger instance."""
    global app_logger
    if app_logger is None:
        app_logger = setup_application_logging()
    return app_logger