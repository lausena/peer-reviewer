"""
LaTeX generation module for creating formatted peer review documents.

This module handles the generation of professional LaTeX documents from peer review data,
including proper formatting, academic styling, and template management.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from jinja2 import Environment, FileSystemLoader, Template
from loguru import logger


@dataclass
class ReviewData:
    """Structured data for a peer review."""
    paper_title: str
    summary: str
    strengths: str
    weaknesses: str
    technical_quality: str
    novelty_significance: str
    clarity_presentation: str
    specific_comments: str
    recommendation: str
    minor_issues: str = ""
    reasoning: str = ""


@dataclass
class DocumentMetadata:
    """Metadata for the review document."""
    review_date: str
    model_name: str
    reasoning_level: str
    review_type: str
    extraction_method: str = ""
    word_count: int = 0
    page_count: int = 0
    temperature: float = 0.7
    max_tokens: int = 4096


class LaTeXGenerator:
    """
    Generator for creating LaTeX formatted peer review documents.
    
    This class handles template loading, data formatting, and LaTeX generation
    with proper academic styling and structure.
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize the LaTeX generator.
        
        Args:
            template_dir: Directory containing LaTeX templates
        """
        if template_dir is None:
            # Use the default template directory relative to this file
            template_dir = Path(__file__).parent.parent / "templates"
        
        self.template_dir = Path(template_dir)
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=False,  # LaTeX has its own escaping rules
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        logger.info(f"LaTeX generator initialized with template dir: {self.template_dir}")
    
    def generate_review_document(
        self,
        review_data: ReviewData,
        metadata: DocumentMetadata,
        output_path: Path,
        template_name: str = "peer_review.tex"
    ) -> bool:
        """
        Generate a complete LaTeX peer review document.
        
        Args:
            review_data: Structured peer review content
            metadata: Document metadata and configuration
            output_path: Path where the LaTeX file should be saved
            template_name: Name of the LaTeX template to use
            
        Returns:
            True if generation was successful, False otherwise
        """
        try:
            logger.info(f"Generating LaTeX document: {output_path.name}")
            
            # Load the template
            template = self.jinja_env.get_template(template_name)
            
            # Prepare template variables
            template_vars = self._prepare_template_variables(review_data, metadata)
            
            # Render the template
            latex_content = template.render(**template_vars)
            
            # Clean and format the LaTeX content
            latex_content = self._clean_latex_content(latex_content)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the LaTeX file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            logger.success(f"LaTeX document generated successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate LaTeX document: {e}")
            return False
    
    def generate_from_model_output(
        self,
        model_output: Dict[str, Any],
        paper_title: str,
        output_path: Path,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Generate LaTeX document from raw model output.
        
        Args:
            model_output: Output from the model interface
            paper_title: Title of the reviewed paper
            output_path: Path where the LaTeX file should be saved
            additional_metadata: Additional metadata to include
            
        Returns:
            True if generation was successful, False otherwise
        """
        try:
            # Extract review data from model output
            review_dict = model_output.get("review", {})
            model_metadata = model_output.get("metadata", {})
            
            # Create ReviewData object
            review_data = ReviewData(
                paper_title=paper_title,
                summary=review_dict.get("summary", "No summary available"),
                strengths=review_dict.get("strengths", "No strengths identified"),
                weaknesses=review_dict.get("weaknesses", "No weaknesses identified"),
                technical_quality=review_dict.get("technical_quality", "Technical quality assessment not available"),
                novelty_significance=review_dict.get("novelty_significance", "Novelty and significance assessment not available"),
                clarity_presentation=review_dict.get("clarity_presentation", "Clarity and presentation assessment not available"),
                specific_comments=review_dict.get("specific_comments", "No specific comments provided"),
                recommendation=review_dict.get("recommendation", "No recommendation provided"),
                minor_issues=review_dict.get("minor_issues", ""),
                reasoning=review_dict.get("reasoning", "")
            )
            
            # Create DocumentMetadata object
            metadata = DocumentMetadata(
                review_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name=model_metadata.get("model_name", "Unknown"),
                reasoning_level=model_metadata.get("reasoning_level", "Unknown"),
                review_type=model_metadata.get("review_type", "comprehensive"),
                extraction_method=additional_metadata.get("extraction_method", "") if additional_metadata else "",
                word_count=additional_metadata.get("word_count", 0) if additional_metadata else 0,
                page_count=additional_metadata.get("page_count", 0) if additional_metadata else 0,
            )
            
            # Add generation config if available
            if "generation_config" in model_metadata and model_metadata["generation_config"]:
                gen_config = model_metadata["generation_config"]
                metadata.temperature = gen_config.get("temperature", 0.7)
                metadata.max_tokens = gen_config.get("max_new_tokens", 4096)
            
            return self.generate_review_document(review_data, metadata, output_path)
            
        except Exception as e:
            logger.error(f"Failed to generate LaTeX from model output: {e}")
            return False
    
    def _prepare_template_variables(
        self,
        review_data: ReviewData,
        metadata: DocumentMetadata
    ) -> Dict[str, Any]:
        """Prepare variables for template rendering."""
        
        # Escape LaTeX special characters in text content
        escaped_review = self._escape_latex_text(review_data)
        
        # Create short title for headers (max 50 chars)
        paper_title_short = review_data.paper_title
        if len(paper_title_short) > 50:
            paper_title_short = paper_title_short[:47] + "..."
        
        template_vars = {
            # Review content (escaped)
            "paper_title": escaped_review.paper_title,
            "paper_title_short": self._escape_latex_special(paper_title_short),
            "summary": self._format_text_content(escaped_review.summary),
            "strengths": self._format_text_content(escaped_review.strengths),
            "weaknesses": self._format_text_content(escaped_review.weaknesses),
            "technical_quality": self._format_text_content(escaped_review.technical_quality),
            "novelty_significance": self._format_text_content(escaped_review.novelty_significance),
            "clarity_presentation": self._format_text_content(escaped_review.clarity_presentation),
            "specific_comments": self._format_text_content(escaped_review.specific_comments),
            "recommendation": self._format_text_content(escaped_review.recommendation),
            "minor_issues": self._format_text_content(escaped_review.minor_issues),
            "reasoning": self._format_text_content(escaped_review.reasoning),
            
            # Metadata
            "review_date": metadata.review_date,
            "model_name": self._escape_latex_special(metadata.model_name),
            "reasoning_level": metadata.reasoning_level,
            "review_type": metadata.review_type.title(),
            "extraction_method": metadata.extraction_method,
            "word_count": f"{metadata.word_count:,}",
            "page_count": str(metadata.page_count),
            "temperature": str(metadata.temperature),
            "max_tokens": f"{metadata.max_tokens:,}",
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        return template_vars
    
    def _escape_latex_text(self, review_data: ReviewData) -> ReviewData:
        """Escape LaTeX special characters in all text fields."""
        return ReviewData(
            paper_title=self._escape_latex_special(review_data.paper_title),
            summary=self._escape_latex_special(review_data.summary),
            strengths=self._escape_latex_special(review_data.strengths),
            weaknesses=self._escape_latex_special(review_data.weaknesses),
            technical_quality=self._escape_latex_special(review_data.technical_quality),
            novelty_significance=self._escape_latex_special(review_data.novelty_significance),
            clarity_presentation=self._escape_latex_special(review_data.clarity_presentation),
            specific_comments=self._escape_latex_special(review_data.specific_comments),
            recommendation=self._escape_latex_special(review_data.recommendation),
            minor_issues=self._escape_latex_special(review_data.minor_issues),
            reasoning=self._escape_latex_special(review_data.reasoning),
        )
    
    def _escape_latex_special(self, text: str) -> str:
        """Escape special LaTeX characters."""
        if not text:
            return ""
        
        # LaTeX special characters and their escaped versions
        latex_chars = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '^': r'\textasciicircum{}',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '\\': r'\textbackslash{}',
        }
        
        escaped_text = text
        for char, escape in latex_chars.items():
            escaped_text = escaped_text.replace(char, escape)
        
        return escaped_text
    
    def _format_text_content(self, text: str) -> str:
        """Format text content for better LaTeX presentation."""
        if not text:
            return "Content not available."
        
        # Convert common markdown-style formatting to LaTeX
        # Bold text
        text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
        text = re.sub(r'__(.*?)__', r'\\textbf{\1}', text)
        
        # Italic text
        text = re.sub(r'\*(.*?)\*', r'\\textit{\1}', text)
        text = re.sub(r'_(.*?)_', r'\\textit{\1}', text)
        
        # Convert bullet points to LaTeX itemize
        if '•' in text or text.strip().startswith('- '):
            lines = text.split('\n')
            formatted_lines = []
            in_list = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('- ') or line.startswith('• '):
                    if not in_list:
                        formatted_lines.append('\\begin{itemize}')
                        in_list = True
                    item_text = line[2:].strip()  # Remove bullet and whitespace
                    formatted_lines.append(f'\\item {item_text}')
                else:
                    if in_list:
                        formatted_lines.append('\\end{itemize}')
                        in_list = False
                    if line:  # Only add non-empty lines
                        formatted_lines.append(line)
            
            if in_list:
                formatted_lines.append('\\end{itemize}')
            
            text = '\n'.join(formatted_lines)
        
        # Ensure proper paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text
    
    def _clean_latex_content(self, latex_content: str) -> str:
        """Clean and format the final LaTeX content."""
        # Remove excessive whitespace
        latex_content = re.sub(r'\n{3,}', '\n\n', latex_content)
        
        # Ensure proper line endings
        latex_content = latex_content.replace('\r\n', '\n').replace('\r', '\n')
        
        return latex_content.strip() + '\n'
    
    def get_available_templates(self) -> list:
        """Get a list of available LaTeX templates."""
        try:
            templates = []
            for file_path in self.template_dir.glob("*.tex"):
                templates.append(file_path.name)
            return sorted(templates)
        except Exception as e:
            logger.error(f"Failed to list templates: {e}")
            return []
    
    def validate_template(self, template_name: str) -> bool:
        """Validate that a template exists and is readable."""
        try:
            template_path = self.template_dir / template_name
            return template_path.exists() and template_path.is_file()
        except Exception:
            return False