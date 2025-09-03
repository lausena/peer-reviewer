"""
Configuration management for the peer reviewer application.

This module handles application configuration, settings persistence,
and environment variable management with proper validation.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class ModelSettings(BaseModel):
    """Model configuration settings."""
    model_name: str = Field(
        default="microsoft/gpt-oss-20b",
        description="Name of the model to use"
    )
    torch_dtype: str = Field(
        default="auto",
        description="PyTorch data type for model weights"
    )
    device_map: str = Field(
        default="auto",
        description="Device mapping strategy for model placement"
    )
    reasoning_level: str = Field(
        default="high",
        description="Reasoning level for model inference"
    )
    max_new_tokens: int = Field(
        default=4096,
        ge=100,
        le=8192,
        description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling parameter"
    )
    
    @validator("reasoning_level")
    def validate_reasoning_level(cls, v):
        """Validate reasoning level values."""
        valid_levels = ["low", "medium", "high"]
        if v.lower() not in valid_levels:
            raise ValueError(f"Reasoning level must be one of: {valid_levels}")
        return v.lower()
    
    @validator("model_name")
    def validate_model_name(cls, v):
        """Validate model name format."""
        if not v or "/" not in v:
            raise ValueError("Model name must be in format 'organization/model-name'")
        return v


class ProcessingSettings(BaseModel):
    """PDF processing and output settings."""
    papers_dir: str = Field(
        default="papers",
        description="Directory containing papers to review"
    )
    reviews_dir: str = Field(
        default="reviews",
        description="Directory for generated reviews"
    )
    min_text_length: int = Field(
        default=1000,
        ge=100,
        description="Minimum text length required for processing"
    )
    default_review_type: str = Field(
        default="comprehensive",
        description="Default type of review to generate"
    )
    latex_template: str = Field(
        default="peer_review.tex",
        description="LaTeX template to use for reviews"
    )
    
    @validator("default_review_type")
    def validate_review_type(cls, v):
        """Validate review type values."""
        valid_types = ["comprehensive", "brief", "technical"]
        if v.lower() not in valid_types:
            raise ValueError(f"Review type must be one of: {valid_types}")
        return v.lower()


class ApplicationSettings(BaseSettings):
    """Main application settings with environment variable support."""
    
    # Model settings
    model: ModelSettings = Field(default_factory=ModelSettings)
    
    # Processing settings
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    
    # Application settings
    verbose_logging: bool = Field(
        default=False,
        description="Enable verbose logging"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    config_file: Optional[str] = Field(
        default=None,
        description="Path to configuration file"
    )
    
    class Config:
        env_prefix = "PEER_REVIEWER_"
        env_nested_delimiter = "__"
        case_sensitive = False
        
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()


class Config:
    """
    Configuration manager for the peer reviewer application.
    
    Handles loading, saving, and validation of application configuration
    with support for environment variables and configuration files.
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = config_file or self._get_default_config_path()
        self.settings: Optional[ApplicationSettings] = None
        self._load_config()
    
    def _get_default_config_path(self) -> Path:
        """Get the default configuration file path."""
        # Use XDG config directory or fallback to home directory
        if os.name == 'posix':  # Unix-like systems
            config_dir = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
        else:  # Windows
            config_dir = Path(os.environ.get('APPDATA', Path.home()))
        
        return config_dir / 'peer-reviewer' / 'config.json'
    
    def _load_config(self) -> None:
        """Load configuration from file and environment variables."""
        try:
            # Load from file if it exists
            config_data = {}
            if self.config_file.exists():
                logger.info(f"Loading configuration from: {self.config_file}")
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            
            # Create settings with file data and environment variables
            self.settings = ApplicationSettings(**config_data)
            
            if not self.config_file.exists():
                # Save default configuration
                self.save_config()
                logger.info(f"Created default configuration: {self.config_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")
            logger.info("Using default configuration")
            self.settings = ApplicationSettings()
    
    def save_config(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.settings:
                logger.error("No settings to save")
                return False
            
            # Ensure config directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert settings to dictionary
            config_dict = self.settings.dict()
            
            # Save to file with pretty formatting
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        if not self.settings:
            return {}
        return self.settings.dict()
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.settings:
                self.settings = ApplicationSettings()
            
            # Handle nested updates
            config_dict = self.settings.dict()
            
            for key, value in updates.items():
                if "." in key:
                    # Handle nested keys like "model.temperature"
                    parts = key.split(".")
                    current = config_dict
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    config_dict[key] = value
            
            # Recreate settings with validation
            self.settings = ApplicationSettings(**config_dict)
            
            # Save updated configuration
            return self.save_config()
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """
        Reset configuration to default values.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.settings = ApplicationSettings()
            return self.save_config()
        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}")
            return False
    
    def get_model_config(self):
        """Get model configuration for use with ModelInterface."""
        if not self.settings:
            return None
        
        from ..core.model_interface import ModelConfig
        
        model_settings = self.settings.model
        return ModelConfig(
            model_name=model_settings.model_name,
            torch_dtype=model_settings.torch_dtype,
            device_map=model_settings.device_map,
            reasoning_level=model_settings.reasoning_level,
            max_new_tokens=model_settings.max_new_tokens,
            temperature=model_settings.temperature,
            top_p=model_settings.top_p,
        )
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate current configuration.
        
        Returns:
            Dictionary with validation results
        """
        try:
            if not self.settings:
                return {"valid": False, "error": "No configuration loaded"}
            
            # Check if directories exist or can be created
            papers_dir = Path(self.settings.processing.papers_dir)
            reviews_dir = Path(self.settings.processing.reviews_dir)
            
            issues = []
            
            # Check papers directory
            if not papers_dir.exists():
                try:
                    papers_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create papers directory: {e}")
            
            # Check reviews directory
            if not reviews_dir.exists():
                try:
                    reviews_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create reviews directory: {e}")
            
            # Check template file
            from ..core.latex_generator import LaTeXGenerator
            latex_gen = LaTeXGenerator()
            if not latex_gen.validate_template(self.settings.processing.latex_template):
                issues.append(f"LaTeX template not found: {self.settings.processing.latex_template}")
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "config": self.get_config()
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Configuration validation failed: {e}"
            }
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a specific setting value.
        
        Args:
            key: Setting key (supports dot notation like "model.temperature")
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        try:
            if not self.settings:
                return default
            
            config_dict = self.settings.dict()
            
            if "." in key:
                parts = key.split(".")
                current = config_dict
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        return default
                return current
            else:
                return config_dict.get(key, default)
                
        except Exception:
            return default
    
    def set_setting(self, key: str, value: Any) -> bool:
        """
        Set a specific setting value.
        
        Args:
            key: Setting key (supports dot notation)
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_config({key: value})