"""
Main peer reviewer orchestration module.

This module coordinates the entire peer review process, from PDF processing
to model inference and LaTeX generation, providing a unified interface
for the complete workflow.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib

from loguru import logger

from .pdf_processor import PDFProcessor
from .model_interface import ModelInterface, ModelConfig
from .latex_generator import LaTeXGenerator


class PeerReviewer:
    """
    Main orchestrator for the peer review process.
    
    This class coordinates PDF processing, AI model inference, and LaTeX generation
    to produce complete peer reviews of research papers.
    """
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        papers_dir: Optional[Path] = None,
        reviews_dir: Optional[Path] = None,
        min_text_length: int = 1000
    ):
        """
        Initialize the peer reviewer.
        
        Args:
            model_config: Configuration for the AI model
            papers_dir: Directory containing papers to review
            reviews_dir: Directory to save generated reviews
            min_text_length: Minimum text length required for processing
        """
        self.model_config = model_config or ModelConfig()
        self.papers_dir = Path(papers_dir) if papers_dir else Path("papers")
        self.reviews_dir = Path(reviews_dir) if reviews_dir else Path("reviews")
        self.min_text_length = min_text_length
        
        # Initialize components
        self.pdf_processor = PDFProcessor(min_text_length=min_text_length)
        self.model_interface = ModelInterface(config=self.model_config)
        self.latex_generator = LaTeXGenerator()
        
        # Ensure directories exist
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.reviews_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Peer reviewer initialized")
        logger.info(f"Papers directory: {self.papers_dir}")
        logger.info(f"Reviews directory: {self.reviews_dir}")
    
    def review_paper(
        self,
        paper_path: Path,
        review_type: str = "comprehensive",
        force_overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a complete peer review for a single paper.
        
        Args:
            paper_path: Path to the PDF paper
            review_type: Type of review to generate
            force_overwrite: Whether to overwrite existing reviews
            
        Returns:
            Dictionary containing the review results and metadata
        """
        logger.info(f"Starting review of: {paper_path.name}")
        
        try:
            # Check if review already exists
            output_path = self._get_output_path(paper_path)
            if output_path.exists() and not force_overwrite:
                logger.info(f"Review already exists: {output_path.name}")
                return {
                    "success": True,
                    "output_path": output_path,
                    "status": "skipped",
                    "message": "Review already exists (use force_overwrite=True to regenerate)"
                }
            
            # Step 1: Extract text from PDF
            logger.info("Extracting text from PDF...")
            pdf_result = self.pdf_processor.extract_text(paper_path)
            
            if not pdf_result["success"] or len(pdf_result["text"]) < self.min_text_length:
                return {
                    "success": False,
                    "error": f"Failed to extract sufficient text from PDF (got {len(pdf_result.get('text', ''))} characters)"
                }
            
            paper_text = pdf_result["text"]
            structured_content = pdf_result["structured_content"]
            extraction_metadata = pdf_result["metadata"]
            
            logger.success(f"Extracted {len(paper_text)} characters from PDF")
            
            # Step 2: Generate peer review using AI model
            logger.info("Generating peer review with AI model...")
            
            # Use structured content if available
            paper_title = structured_content.get("title", paper_path.stem)
            paper_abstract = structured_content.get("abstract", "")
            
            model_result = self.model_interface.generate_peer_review(
                paper_text=paper_text,
                paper_title=paper_title,
                paper_abstract=paper_abstract,
                review_type=review_type
            )
            
            if not model_result["success"]:
                return {
                    "success": False,
                    "error": f"Model inference failed: {model_result.get('error', 'Unknown error')}"
                }
            
            logger.success("AI peer review generated successfully")
            
            # Step 3: Generate LaTeX document
            logger.info("Generating LaTeX document...")
            
            # Prepare additional metadata
            additional_metadata = {
                "extraction_method": pdf_result["extraction_method"],
                "word_count": len(paper_text.split()),
                "page_count": extraction_metadata.page_count,
                "processing_date": datetime.now().isoformat(),
                "paper_file": paper_path.name
            }
            
            latex_success = self.latex_generator.generate_from_model_output(
                model_output=model_result,
                paper_title=paper_title,
                output_path=output_path,
                additional_metadata=additional_metadata
            )
            
            if not latex_success:
                return {
                    "success": False,
                    "error": "Failed to generate LaTeX document"
                }
            
            logger.success(f"LaTeX document generated: {output_path}")
            
            # Generate processing summary
            processing_summary = {
                "paper_path": str(paper_path),
                "output_path": str(output_path),
                "review_type": review_type,
                "processing_time": datetime.now().isoformat(),
                "text_length": len(paper_text),
                "word_count": len(paper_text.split()),
                "page_count": extraction_metadata.page_count,
                "extraction_method": pdf_result["extraction_method"],
                "model_name": self.model_config.model_name,
                "reasoning_level": self.model_config.reasoning_level
            }
            
            return {
                "success": True,
                "output_path": output_path,
                "status": "completed",
                "summary": processing_summary,
                "pdf_result": pdf_result,
                "model_result": model_result
            }
            
        except Exception as e:
            logger.error(f"Review failed for {paper_path.name}: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def review_all_papers(
        self,
        review_type: str = "comprehensive",
        force_overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Review all PDF papers in the papers directory.
        
        Args:
            review_type: Type of review to generate
            force_overwrite: Whether to overwrite existing reviews
            
        Returns:
            Dictionary containing results for all papers
        """
        logger.info("Starting batch review of all papers")
        
        # Find all PDF files
        pdf_files = list(self.papers_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.papers_dir}")
            return {
                "success": True,
                "message": "No PDF files found to process",
                "results": []
            }
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each paper
        results = []
        successful = 0
        failed = 0
        
        for pdf_path in pdf_files:
            logger.info(f"Processing {pdf_path.name} ({successful + failed + 1}/{len(pdf_files)})")
            
            try:
                result = self.review_paper(
                    paper_path=pdf_path,
                    review_type=review_type,
                    force_overwrite=force_overwrite
                )
                
                results.append({
                    "paper": pdf_path.name,
                    "result": result
                })
                
                if result["success"]:
                    successful += 1
                    logger.success(f"Completed: {pdf_path.name}")
                else:
                    failed += 1
                    logger.error(f"Failed: {pdf_path.name} - {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                failed += 1
                logger.error(f"Unexpected error processing {pdf_path.name}: {e}")
                results.append({
                    "paper": pdf_path.name,
                    "result": {"success": False, "error": str(e)}
                })
        
        # Summary
        logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
        
        return {
            "success": True,
            "total": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "results": results
        }
    
    def get_paper_info(self, paper_path: Path) -> Dict[str, Any]:
        """
        Get detailed information about a paper without generating a review.
        
        Args:
            paper_path: Path to the PDF paper
            
        Returns:
            Dictionary containing paper information and statistics
        """
        try:
            # Get basic document stats
            stats = self.pdf_processor.get_document_stats(paper_path)
            
            # Check if review exists
            output_path = self._get_output_path(paper_path)
            stats["review_exists"] = output_path.exists()
            stats["review_path"] = str(output_path) if output_path.exists() else None
            
            return {
                "success": True,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get paper info for {paper_path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_papers(self) -> Dict[str, Any]:
        """
        List all available papers and their status.
        
        Returns:
            Dictionary containing paper list and status information
        """
        try:
            pdf_files = list(self.papers_dir.glob("*.pdf"))
            
            papers_info = []
            for pdf_path in sorted(pdf_files):
                info = self.get_paper_info(pdf_path)
                if info["success"]:
                    papers_info.append({
                        "path": str(pdf_path),
                        "name": pdf_path.name,
                        "stats": info["stats"]
                    })
            
            return {
                "success": True,
                "papers_directory": str(self.papers_dir),
                "reviews_directory": str(self.reviews_dir),
                "total_papers": len(papers_info),
                "papers": papers_info
            }
            
        except Exception as e:
            logger.error(f"Failed to list papers: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_output_path(self, paper_path: Path) -> Path:
        """Generate the output path for a review file."""
        # Create a safe filename based on the paper name
        safe_name = self._sanitize_filename(paper_path.stem)
        return self.reviews_dir / f"{safe_name}_review.tex"
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for cross-platform compatibility."""
        # Remove or replace problematic characters
        import re
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        safe_filename = re.sub(r'\s+', '_', safe_filename)
        safe_filename = safe_filename.strip('._')
        
        # Ensure the filename isn't too long
        if len(safe_filename) > 100:
            safe_filename = safe_filename[:100]
        
        return safe_filename or "unnamed_paper"
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status information about the AI model."""
        return self.model_interface.get_model_info()
    
    def cleanup(self) -> None:
        """Clean up resources, particularly the AI model."""
        logger.info("Cleaning up resources...")
        try:
            self.model_interface.unload_model()
            logger.success("Resources cleaned up successfully")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()