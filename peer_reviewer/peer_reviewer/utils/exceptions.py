"""
Custom exceptions for the peer reviewer application.

This module defines specific exception classes for different error conditions
that can occur during the peer review process, providing better error handling
and debugging capabilities.
"""

from typing import Optional, Dict, Any


class PeerReviewerException(Exception):
    """Base exception class for all peer reviewer errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception with message and optional details.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class PDFProcessingError(PeerReviewerException):
    """Exception raised when PDF processing fails."""
    
    def __init__(
        self,
        message: str,
        pdf_path: Optional[str] = None,
        extraction_method: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize PDF processing error.
        
        Args:
            message: Error message
            pdf_path: Path to the PDF file that failed
            extraction_method: Method used for text extraction
            details: Additional error details
        """
        error_details = details or {}
        if pdf_path:
            error_details["pdf_path"] = pdf_path
        if extraction_method:
            error_details["extraction_method"] = extraction_method
        
        super().__init__(message, error_details)
        self.pdf_path = pdf_path
        self.extraction_method = extraction_method


class ModelLoadError(PeerReviewerException):
    """Exception raised when model loading fails."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize model loading error.
        
        Args:
            message: Error message
            model_name: Name of the model that failed to load
            error_type: Type of error (memory, network, etc.)
            details: Additional error details
        """
        error_details = details or {}
        if model_name:
            error_details["model_name"] = model_name
        if error_type:
            error_details["error_type"] = error_type
        
        super().__init__(message, error_details)
        self.model_name = model_name
        self.error_type = error_type


class ModelInferenceError(PeerReviewerException):
    """Exception raised when model inference fails."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        input_length: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize model inference error.
        
        Args:
            message: Error message
            model_name: Name of the model used
            input_length: Length of input text in characters
            details: Additional error details
        """
        error_details = details or {}
        if model_name:
            error_details["model_name"] = model_name
        if input_length:
            error_details["input_length"] = input_length
        
        super().__init__(message, error_details)
        self.model_name = model_name
        self.input_length = input_length


class LaTeXGenerationError(PeerReviewerException):
    """Exception raised when LaTeX generation fails."""
    
    def __init__(
        self,
        message: str,
        template_name: Optional[str] = None,
        output_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LaTeX generation error.
        
        Args:
            message: Error message
            template_name: Name of the LaTeX template used
            output_path: Intended output path
            details: Additional error details
        """
        error_details = details or {}
        if template_name:
            error_details["template_name"] = template_name
        if output_path:
            error_details["output_path"] = output_path
        
        super().__init__(message, error_details)
        self.template_name = template_name
        self.output_path = output_path


class ConfigurationError(PeerReviewerException):
    """Exception raised when configuration is invalid or missing."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
            config_file: Path to configuration file
            details: Additional error details
        """
        error_details = details or {}
        if config_key:
            error_details["config_key"] = config_key
        if config_file:
            error_details["config_file"] = config_file
        
        super().__init__(message, error_details)
        self.config_key = config_key
        self.config_file = config_file


class ValidationError(PeerReviewerException):
    """Exception raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field_name: Name of the field that failed validation
            field_value: Value that failed validation
            expected_type: Expected type or format
            details: Additional error details
        """
        error_details = details or {}
        if field_name:
            error_details["field_name"] = field_name
        if field_value is not None:
            error_details["field_value"] = str(field_value)
        if expected_type:
            error_details["expected_type"] = expected_type
        
        super().__init__(message, error_details)
        self.field_name = field_name
        self.field_value = field_value
        self.expected_type = expected_type


class ResourceError(PeerReviewerException):
    """Exception raised when system resources are insufficient."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        required_amount: Optional[str] = None,
        available_amount: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize resource error.
        
        Args:
            message: Error message
            resource_type: Type of resource (memory, disk, etc.)
            required_amount: Required amount of resource
            available_amount: Available amount of resource
            details: Additional error details
        """
        error_details = details or {}
        if resource_type:
            error_details["resource_type"] = resource_type
        if required_amount:
            error_details["required_amount"] = required_amount
        if available_amount:
            error_details["available_amount"] = available_amount
        
        super().__init__(message, error_details)
        self.resource_type = resource_type
        self.required_amount = required_amount
        self.available_amount = available_amount


class FileSystemError(PeerReviewerException):
    """Exception raised when file system operations fail."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        permissions: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize file system error.
        
        Args:
            message: Error message
            file_path: Path to the file that caused the error
            operation: Type of operation (read, write, create, etc.)
            permissions: Required permissions
            details: Additional error details
        """
        error_details = details or {}
        if file_path:
            error_details["file_path"] = file_path
        if operation:
            error_details["operation"] = operation
        if permissions:
            error_details["permissions"] = permissions
        
        super().__init__(message, error_details)
        self.file_path = file_path
        self.operation = operation
        self.permissions = permissions


def handle_exception(exception: Exception, context: str = "") -> PeerReviewerException:
    """
    Convert generic exceptions to appropriate PeerReviewerException types.
    
    Args:
        exception: The original exception
        context: Additional context about where the exception occurred
        
    Returns:
        Appropriate PeerReviewerException subclass
    """
    error_message = str(exception)
    if context:
        error_message = f"{context}: {error_message}"
    
    # Map common exceptions to specific types
    if isinstance(exception, FileNotFoundError):
        return FileSystemError(
            error_message,
            file_path=getattr(exception, 'filename', None),
            operation="read",
            details={"original_exception": type(exception).__name__}
        )
    
    elif isinstance(exception, PermissionError):
        return FileSystemError(
            error_message,
            file_path=getattr(exception, 'filename', None),
            operation="access",
            permissions="insufficient",
            details={"original_exception": type(exception).__name__}
        )
    
    elif isinstance(exception, MemoryError):
        return ResourceError(
            error_message,
            resource_type="memory",
            details={"original_exception": type(exception).__name__}
        )
    
    elif isinstance(exception, ValueError):
        return ValidationError(
            error_message,
            details={"original_exception": type(exception).__name__}
        )
    
    elif isinstance(exception, ImportError):
        return ConfigurationError(
            error_message,
            details={
                "original_exception": type(exception).__name__,
                "suggestion": "Check if all required dependencies are installed"
            }
        )
    
    else:
        # Generic exception wrapper
        return PeerReviewerException(
            error_message,
            details={
                "original_exception": type(exception).__name__,
                "context": context
            }
        )