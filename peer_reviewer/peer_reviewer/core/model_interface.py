"""
Model interface for OpenAI's gpt-oss-20b model with harmony response format.

This module provides a clean interface to the gpt-oss-20b model for generating
peer reviews with proper reasoning capabilities and optimized settings for M1 Max.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from loguru import logger


@dataclass
class ModelConfig:
    """Configuration for the gpt-oss-20b model."""
    model_name: str = "microsoft/gpt-oss-20b"
    torch_dtype: str = "auto"
    device_map: str = "auto"
    reasoning_level: str = "high"
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    use_cache: bool = True
    pad_token_id: Optional[int] = None


class ModelInterface:
    """
    Interface for OpenAI's gpt-oss-20b model with harmony response format.
    
    This class handles model loading, prompt formatting, and inference
    with optimization for M1 Max architecture and peer review generation.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the model interface.
        
        Args:
            config: Model configuration options
        """
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self._model_loaded = False
        
        logger.info(f"Model interface initialized with config: {self.config}")
    
    def load_model(self) -> None:
        """
        Load the gpt-oss-20b model and tokenizer.
        
        Optimized for M1 Max with 64GB RAM using torch_dtype="auto" and device_map="auto"
        """
        if self._model_loaded:
            logger.info("Model already loaded")
            return
        
        logger.info(f"Loading model: {self.config.model_name}")
        logger.info("This may take several minutes for the first load...")
        
        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Set padding token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.config.pad_token_id = self.tokenizer.eos_token_id
            
            # Load model with M1 Max optimization
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.torch_dtype == "auto" else getattr(torch, self.config.torch_dtype),
                device_map=self.config.device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Configure generation parameters
            self.generation_config = GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                use_cache=self.config.use_cache,
                pad_token_id=self.config.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            self._model_loaded = True
            logger.success("Model loaded successfully")
            
            # Log model info
            if hasattr(self.model, 'config'):
                logger.info(f"Model parameters: ~{self.model.num_parameters() / 1e9:.1f}B")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def generate_peer_review(
        self,
        paper_text: str,
        paper_title: str = "",
        paper_abstract: str = "",
        review_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate a peer review for the given paper.
        
        Args:
            paper_text: Full text of the research paper
            paper_title: Title of the paper (if available)
            paper_abstract: Abstract of the paper (if available)
            review_type: Type of review to generate (comprehensive, brief, technical)
            
        Returns:
            Dictionary containing the generated review and metadata
        """
        if not self._model_loaded:
            self.load_model()
        
        logger.info(f"Generating {review_type} peer review")
        logger.info(f"Paper title: {paper_title[:100]}..." if paper_title else "No title provided")
        
        # Create the peer review prompt
        prompt = self._create_peer_review_prompt(
            paper_text, paper_title, paper_abstract, review_type
        )
        
        try:
            # Generate the review
            response = self._generate_text(prompt)
            
            # Parse the harmony response format
            review_data = self._parse_harmony_response(response)
            
            return {
                "review": review_data,
                "metadata": {
                    "paper_title": paper_title,
                    "review_type": review_type,
                    "model_name": self.config.model_name,
                    "reasoning_level": self.config.reasoning_level,
                    "generation_config": self.generation_config.to_dict() if self.generation_config else None
                },
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to generate peer review: {e}")
            return {
                "review": None,
                "error": str(e),
                "success": False
            }
    
    def _create_peer_review_prompt(
        self,
        paper_text: str,
        paper_title: str,
        paper_abstract: str,
        review_type: str
    ) -> str:
        """Create the prompt for peer review generation."""
        
        # System prompt with reasoning level
        system_prompt = f"""Reasoning: {self.config.reasoning_level}

You are an expert academic peer reviewer with extensive experience in evaluating research papers across multiple disciplines. Your task is to provide a comprehensive, constructive, and rigorous peer review that follows academic standards.

Please analyze the following research paper and provide a detailed peer review that includes:

1. **Summary**: A concise summary of the paper's main contributions and findings
2. **Strengths**: Key strengths and positive aspects of the work
3. **Weaknesses**: Areas that need improvement or raise concerns
4. **Technical Quality**: Assessment of methodology, experimental design, and analysis
5. **Novelty and Significance**: Evaluation of the originality and impact of the contributions
6. **Clarity and Presentation**: Quality of writing, organization, and figures
7. **Specific Comments**: Detailed line-by-line or section-by-section feedback
8. **Recommendation**: Overall assessment and recommendation for publication
9. **Minor Issues**: Typos, formatting, and other minor corrections needed

Use the harmony response format and provide thorough reasoning for all assessments. Be constructive, fair, and provide actionable feedback that will help improve the paper."""

        # Paper information section
        paper_info = ""
        if paper_title:
            paper_info += f"**Title**: {paper_title}\n\n"
        if paper_abstract:
            paper_info += f"**Abstract**: {paper_abstract}\n\n"
        
        # Truncate paper text if too long (keep within token limits)
        max_paper_length = 8000  # Adjust based on model's context window
        if len(paper_text) > max_paper_length:
            paper_text = paper_text[:max_paper_length] + "\n\n[Content truncated for length]"
        
        paper_info += f"**Full Paper Text**:\n{paper_text}"
        
        # Combine system prompt and paper content
        full_prompt = f"{system_prompt}\n\n{paper_info}\n\nPlease provide your comprehensive peer review:"
        
        return full_prompt
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text using the loaded model."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length - self.config.max_new_tokens
            )
            
            # Move to appropriate device
            if self.model.device.type != 'cpu':
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            logger.info("Generating model response...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            logger.success(f"Generated response ({len(response)} characters)")
            return response
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")
    
    def _parse_harmony_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the harmony response format into structured data.
        
        The harmony format typically includes reasoning and structured output.
        """
        parsed = {
            "raw_response": response,
            "summary": "",
            "strengths": "",
            "weaknesses": "",
            "technical_quality": "",
            "novelty_significance": "",
            "clarity_presentation": "",
            "specific_comments": "",
            "recommendation": "",
            "minor_issues": "",
            "reasoning": ""
        }
        
        try:
            # Extract different sections using common academic review patterns
            sections = {
                r"(?i)(?:^|\n)\s*(?:\*\*)?summary(?:\*\*)?\s*:?\s*\n(.+?)(?=\n\s*(?:\*\*)?(?:strengths?|weakness)|\n\n|\Z)": "summary",
                r"(?i)(?:^|\n)\s*(?:\*\*)?strengths?(?:\*\*)?\s*:?\s*\n(.+?)(?=\n\s*(?:\*\*)?(?:weakness|technical)|\n\n|\Z)": "strengths",
                r"(?i)(?:^|\n)\s*(?:\*\*)?weakness(?:es)?(?:\*\*)?\s*:?\s*\n(.+?)(?=\n\s*(?:\*\*)?(?:technical|novelty)|\n\n|\Z)": "weaknesses",
                r"(?i)(?:^|\n)\s*(?:\*\*)?technical(?:\s+quality)?(?:\*\*)?\s*:?\s*\n(.+?)(?=\n\s*(?:\*\*)?(?:novelty|clarity)|\n\n|\Z)": "technical_quality",
                r"(?i)(?:^|\n)\s*(?:\*\*)?novelty(?:\s+and\s+significance)?(?:\*\*)?\s*:?\s*\n(.+?)(?=\n\s*(?:\*\*)?(?:clarity|specific)|\n\n|\Z)": "novelty_significance",
                r"(?i)(?:^|\n)\s*(?:\*\*)?clarity(?:\s+and\s+presentation)?(?:\*\*)?\s*:?\s*\n(.+?)(?=\n\s*(?:\*\*)?(?:specific|recommendation)|\n\n|\Z)": "clarity_presentation",
                r"(?i)(?:^|\n)\s*(?:\*\*)?specific\s+comments?(?:\*\*)?\s*:?\s*\n(.+?)(?=\n\s*(?:\*\*)?(?:recommendation|minor)|\n\n|\Z)": "specific_comments",
                r"(?i)(?:^|\n)\s*(?:\*\*)?recommendation(?:\*\*)?\s*:?\s*\n(.+?)(?=\n\s*(?:\*\*)?(?:minor|reasoning)|\n\n|\Z)": "recommendation",
                r"(?i)(?:^|\n)\s*(?:\*\*)?minor\s+issues?(?:\*\*)?\s*:?\s*\n(.+?)(?=\n\s*(?:\*\*)?reasoning|\n\n|\Z)": "minor_issues",
            }
            
            for pattern, section_name in sections.items():
                import re
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    parsed[section_name] = match.group(1).strip()
            
            # If structured sections aren't found, use the full response
            if not any(parsed[key] for key in parsed if key != "raw_response"):
                parsed["summary"] = response
            
        except Exception as e:
            logger.warning(f"Failed to parse harmony response structure: {e}")
            parsed["summary"] = response
        
        return parsed
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._model_loaded:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "model_name": self.config.model_name,
            "device": str(self.model.device) if self.model else "unknown",
            "torch_dtype": str(self.model.dtype) if self.model else "unknown",
            "generation_config": self.generation_config.to_dict() if self.generation_config else None,
        }
        
        if hasattr(self.model, 'config'):
            info["parameters"] = f"~{self.model.num_parameters() / 1e9:.1f}B"
            info["model_config"] = {
                "vocab_size": getattr(self.model.config, 'vocab_size', 'unknown'),
                "hidden_size": getattr(self.model.config, 'hidden_size', 'unknown'),
                "num_attention_heads": getattr(self.model.config, 'num_attention_heads', 'unknown'),
                "num_hidden_layers": getattr(self.model.config, 'num_hidden_layers', 'unknown'),
            }
        
        return info
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model_loaded:
            logger.info("Unloading model...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.generation_config = None
            self._model_loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.success("Model unloaded successfully")