"""
Command-line interface for the peer reviewer application.

This module provides a user-friendly CLI using typer with rich output formatting
for running peer reviews on research papers with various options and configurations.
"""

import sys
from pathlib import Path
from typing import Optional, List
import typer
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from loguru import logger

from .core.reviewer import PeerReviewer
from .core.model_interface import ModelConfig
from .utils.config import Config


# Initialize Typer app and Rich console
app = typer.Typer(
    name="peer-reviewer",
    help="AI-powered peer review system for research papers using OpenAI's gpt-oss-20b model",
    add_completion=False,
    rich_markup_mode="markdown"
)
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    logger.remove()  # Remove default handler
    
    if verbose:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="DEBUG"
        )
    else:
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level="INFO"
        )


@app.command()
def review(
    paper_path: Optional[Path] = typer.Argument(
        None,
        help="Path to specific paper to review. If not provided, reviews all papers in papers/ directory"
    ),
    papers_dir: Path = typer.Option(
        Path("papers"),
        "--papers-dir", "-p",
        help="Directory containing papers to review"
    ),
    reviews_dir: Path = typer.Option(
        Path("reviews"),
        "--reviews-dir", "-r", 
        help="Directory to save generated reviews"
    ),
    model_name: str = typer.Option(
        "microsoft/gpt-oss-20b",
        "--model", "-m",
        help="Model name to use for review generation"
    ),
    reasoning_level: str = typer.Option(
        "high",
        "--reasoning", "-l",
        help="Reasoning level (high, medium, low)"
    ),
    review_type: str = typer.Option(
        "comprehensive",
        "--type", "-t",
        help="Type of review (comprehensive, brief, technical)"
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        help="Temperature for text generation (0.0-1.0)"
    ),
    max_tokens: int = typer.Option(
        4096,
        "--max-tokens",
        help="Maximum tokens for generated review"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Overwrite existing review files"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    )
) -> None:
    """
    Generate peer reviews for research papers using AI.
    
    This command processes PDF files and generates comprehensive peer reviews
    using OpenAI's gpt-oss-20b model with configurable parameters.
    
    **Examples:**
    
    - Review all papers: `uv run peer-reviewer`
    - Review specific paper: `uv run peer-reviewer papers/my-paper.pdf`
    - Use different model settings: `uv run peer-reviewer --temperature 0.5 --reasoning medium`
    """
    setup_logging(verbose)
    
    # Print banner
    console.print(Panel.fit(
        "[bold blue]AI Peer Reviewer[/bold blue]\n"
        "Powered by OpenAI's gpt-oss-20b model",
        border_style="blue"
    ))
    
    try:
        # Validate directories
        if not papers_dir.exists():
            rprint(f"[red]Error:[/red] Papers directory not found: {papers_dir}")
            raise typer.Exit(1)
        
        # Create reviews directory if it doesn't exist
        reviews_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure model
        model_config = ModelConfig(
            model_name=model_name,
            reasoning_level=reasoning_level,
            temperature=temperature,
            max_new_tokens=max_tokens
        )
        
        # Initialize reviewer
        reviewer = PeerReviewer(
            model_config=model_config,
            papers_dir=papers_dir,
            reviews_dir=reviews_dir
        )
        
        # Determine papers to process
        if paper_path:
            if not paper_path.exists():
                rprint(f"[red]Error:[/red] Paper file not found: {paper_path}")
                raise typer.Exit(1)
            papers_to_process = [paper_path]
        else:
            papers_to_process = list(papers_dir.glob("*.pdf"))
            if not papers_to_process:
                rprint(f"[yellow]Warning:[/yellow] No PDF files found in {papers_dir}")
                raise typer.Exit(0)
        
        # Display processing plan
        table = Table(title="Processing Plan")
        table.add_column("Paper", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Output", style="blue")
        
        for paper in papers_to_process:
            output_path = reviews_dir / f"{paper.stem}_review.tex"
            status = "Will process" if not output_path.exists() or force else "Exists (use --force to overwrite)"
            table.add_row(paper.name, status, output_path.name)
        
        console.print(table)
        
        # Confirm processing
        if not typer.confirm("\nProceed with processing?"):
            rprint("[yellow]Cancelled by user[/yellow]")
            raise typer.Exit(0)
        
        # Process papers
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            main_task = progress.add_task("Processing papers...", total=len(papers_to_process))
            
            results = []
            for paper_path in papers_to_process:
                progress.update(main_task, description=f"Processing {paper_path.name}...")
                
                try:
                    result = reviewer.review_paper(
                        paper_path,
                        review_type=review_type,
                        force_overwrite=force
                    )
                    results.append((paper_path, result))
                    
                    if result["success"]:
                        rprint(f"[green]✓[/green] Successfully reviewed: {paper_path.name}")
                    else:
                        rprint(f"[red]✗[/red] Failed to review: {paper_path.name}")
                        if "error" in result:
                            rprint(f"  Error: {result['error']}")
                
                except Exception as e:
                    logger.error(f"Unexpected error processing {paper_path}: {e}")
                    rprint(f"[red]✗[/red] Unexpected error: {paper_path.name}")
                    results.append((paper_path, {"success": False, "error": str(e)}))
                
                progress.advance(main_task)
        
        # Summary report
        console.print("\n" + "="*50)
        console.print("[bold]Processing Summary[/bold]")
        
        successful = sum(1 for _, result in results if result["success"])
        failed = len(results) - successful
        
        rprint(f"[green]Successful:[/green] {successful}")
        rprint(f"[red]Failed:[/red] {failed}")
        
        if successful > 0:
            rprint(f"\n[blue]Reviews saved to:[/blue] {reviews_dir}")
        
        # Exit with appropriate code
        sys.exit(0 if failed == 0 else 1)
        
    except KeyboardInterrupt:
        rprint("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        logger.error(f"Application error: {e}")
        rprint(f"[red]Fatal error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def info(
    papers_dir: Path = typer.Option(
        Path("papers"),
        "--papers-dir", "-p",
        help="Directory containing papers"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed information"
    )
) -> None:
    """
    Show information about available papers and system status.
    """
    setup_logging(verbose)
    
    console.print(Panel.fit(
        "[bold cyan]System Information[/bold cyan]",
        border_style="cyan"
    ))
    
    # System info
    import torch
    from transformers import __version__ as transformers_version
    
    info_table = Table(title="System Status")
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Status", style="green")
    
    info_table.add_row("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    info_table.add_row("PyTorch Version", torch.__version__)
    info_table.add_row("Transformers Version", transformers_version)
    info_table.add_row("CUDA Available", "Yes" if torch.cuda.is_available() else "No")
    info_table.add_row("MPS Available", "Yes" if torch.backends.mps.is_available() else "No")
    
    console.print(info_table)
    
    # Papers info
    if papers_dir.exists():
        papers = list(papers_dir.glob("*.pdf"))
        
        if papers:
            papers_table = Table(title=f"Available Papers ({len(papers)})")
            papers_table.add_column("File", style="cyan")
            papers_table.add_column("Size", style="green")
            papers_table.add_column("Modified", style="blue")
            
            for paper in sorted(papers):
                size_mb = paper.stat().st_size / (1024 * 1024)
                import datetime
                modified = datetime.datetime.fromtimestamp(paper.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                papers_table.add_row(paper.name, f"{size_mb:.1f} MB", modified)
            
            console.print(papers_table)
        else:
            rprint(f"[yellow]No PDF files found in {papers_dir}[/yellow]")
    else:
        rprint(f"[red]Papers directory not found: {papers_dir}[/red]")


@app.command()
def config(
    model_name: Optional[str] = typer.Option(None, "--model", help="Default model name"),
    reasoning_level: Optional[str] = typer.Option(None, "--reasoning", help="Default reasoning level"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="Default temperature"),
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    reset: bool = typer.Option(False, "--reset", help="Reset to default configuration")
) -> None:
    """
    Manage application configuration.
    """
    config_manager = Config()
    
    if reset:
        config_manager.reset_to_defaults()
        rprint("[green]Configuration reset to defaults[/green]")
        return
    
    if show or not any([model_name, reasoning_level, temperature]):
        current_config = config_manager.get_config()
        
        config_table = Table(title="Current Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        
        for key, value in current_config.items():
            config_table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(config_table)
        return
    
    # Update configuration
    updates = {}
    if model_name:
        updates["model_name"] = model_name
    if reasoning_level:
        updates["reasoning_level"] = reasoning_level
    if temperature is not None:
        updates["temperature"] = temperature
    
    config_manager.update_config(updates)
    rprint("[green]Configuration updated[/green]")


def main() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()