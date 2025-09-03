"""Interface for gpt-oss-20b model to generate LaTeX peer reviews."""

import torch
import signal
from contextlib import contextmanager
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


class GPTOSSModel:
    """Interface for the gpt-oss-20b model."""
    
    def __init__(self):
        """Initialize the model and tokenizer."""
        self.model_name = "openai/gpt-oss-20b"
        
        # Set memory optimization for M1 Mac
        if torch.backends.mps.is_available():
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        
        # Determine the best available device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        print(f"Loading model {self.model_name} on device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"  # For generation
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with M1 Mac compatibility
        if torch.backends.mps.is_available():
            # M1 Mac with MPS backend
            self.device = torch.device("mps")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,  # More efficient than float32
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_cache=False,  # Reduce memory usage
            )
            self.model = self.model.to(self.device)
        elif self.device.type == "cuda":
            # CUDA GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # CPU fallback
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self.model = self.model.to(self.device)
        
        print("Model loaded successfully!")
    
    def generate_latex_review(self, paper_content: str, max_length: int = 800) -> str:
        """
        Generate a complete LaTeX peer review document.
        
        Args:
            paper_content: The extracted text content from the research paper
            max_length: Maximum length for generation
            
        Returns:
            Complete LaTeX document as a string
        """
        
        # Create the system prompt for high-reasoning LaTeX generation
        system_prompt = """Reasoning: high

You are an expert peer reviewer with deep expertise in academic research evaluation. Generate a complete, publication-ready LaTeX document that provides a comprehensive peer review of the following research paper. 

Your LaTeX document should include:
1. Proper LaTeX document structure with documentclass, packages, and formatting
2. Title page with review information
3. Executive summary section
4. Detailed technical analysis
5. Strengths and contributions section  
6. Weaknesses and limitations section
7. Detailed comments and suggestions for improvement
8. Minor issues (grammar, formatting, references)
9. Overall recommendation with rationale
10. Proper LaTeX formatting, equations, references, and academic style

The document should be self-contained and ready for compilation. Use proper LaTeX syntax throughout.

Research Paper Content:
"""
        
        # Combine system prompt with paper content
        full_prompt = system_prompt + paper_content + "\n\nGenerate the complete LaTeX peer review document:"
        
        # Tokenize input
        inputs = self.tokenizer.encode(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # Leave room for generation
            padding=False
        ).to(self.device)
        
        print(f"Input tokens: {inputs.shape[1]}")
        print("Generating LaTeX review... (optimized, max 2 minutes timeout)")
        
        # Clear cache to free memory
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Generate response with optimized settings for speed
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    temperature=0.9,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.03,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    num_beams=1  # Fastest generation
                )
        except Exception as e:
            print(f"Generation failed: {e}")
            print("Using fallback review template...")
            return self._generate_fallback_review()
        
        # Decode the generated response
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated LaTeX content (after the prompt)
        latex_content = full_output[len(full_prompt):].strip()
        
        print(f"Generated LaTeX document ({len(latex_content)} characters)")
        
        return latex_content
    
    def _generate_fallback_review(self) -> str:
        """Generate a basic LaTeX review template when model generation fails."""
        return r"""
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\geometry{margin=1in}

\title{Peer Review Report}
\author{AI Reviewer}
\date{\today}

\begin{document}
\maketitle

\section{Executive Summary}
This paper presents research that has been analyzed using automated content analysis. Due to computational constraints, this review uses a template-based approach while maintaining academic standards.

\section{Strengths}
\begin{itemize}
\item Novel approach to the research problem
\item Clear problem formulation
\item Adequate experimental setup
\end{itemize}

\section{Weaknesses}
\begin{itemize}
\item Limited evaluation scope
\item Missing comparison with recent work
\item Presentation could be improved
\end{itemize}

\section{Recommendation}
This paper shows promise but requires major revisions before publication.

\end{document}
"""
    
    def _generate_demo_review(self, paper_content: str) -> str:
        """Generate a demo review based on paper content analysis."""
        # Extract basic info about the paper
        word_count = len(paper_content.split())
        has_equations = '\\' in paper_content or '$' in paper_content
        has_figures = 'figure' in paper_content.lower() or 'fig.' in paper_content.lower()
        has_tables = 'table' in paper_content.lower() or 'tab.' in paper_content.lower()
        
        return rf"""
\documentclass[11pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{geometry}}
\geometry{{margin=1in}}
\usepackage{{amsmath}}
\usepackage{{cite}}

\title{{Peer Review Report}}
\author{{AI Reviewer (Demo Mode)}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{Executive Summary}}
This paper contains approximately {word_count} words and presents research findings in a structured format. The work demonstrates {'mathematical formulations' if has_equations else 'theoretical concepts'} with {'visual aids including figures' if has_figures else 'textual explanations'} {'and tabulated data' if has_tables else ''}.

\section{{Strengths}}
\begin{{itemize}}
\item Clear problem formulation and research objectives
\item {'Appropriate use of mathematical notation and equations' if has_equations else 'Well-structured theoretical framework'}
\item {'Good integration of visual elements (figures/tables)' if (has_figures or has_tables) else 'Comprehensive textual analysis'}
\item Adequate scope for the research domain
\end{{itemize}}

\section{{Areas for Improvement}}
\begin{{itemize}}
\item Consider expanding the literature review section
\item Provide more detailed methodology description
\item Include additional experimental validation
\item Improve discussion of limitations
\end{{itemize}}

\section{{Technical Comments}}
\subsection{{Methodology}}
The paper would benefit from more detailed explanation of the experimental setup and data collection procedures.

\subsection{{Results}}
{'The mathematical results appear sound but could benefit from additional validation.' if has_equations else 'The presented results are reasonable but require more comprehensive analysis.'}

\section{{Minor Issues}}
\begin{{itemize}}
\item Check formatting consistency throughout the document
\item Verify all references are properly cited
\item Consider improving figure/table captions for clarity
\end{{itemize}}

\section{{Overall Assessment}}
This paper presents {'technically sound research' if has_equations else 'valuable insights'} that contributes to the field. With the suggested improvements, particularly in methodology description and validation, this work has potential for publication.

\textbf{{Recommendation:}} Minor to moderate revisions required.

\end{{document}}
"""


# Global model instance (lazy loading)
_model_instance = None


def get_model() -> GPTOSSModel:
    """Get the singleton model instance (lazy loading)."""
    global _model_instance
    if _model_instance is None:
        _model_instance = GPTOSSModel()
    return _model_instance


@contextmanager
def timeout_context(seconds):
    """Context manager for timeout functionality."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def generate_latex_review(paper_content: str) -> str:
    """
    Generate a LaTeX peer review using optimized gpt-oss-20b model.
    
    Args:
        paper_content: The extracted text content from the research paper
        
    Returns:
        Complete LaTeX document as a string
    """
    model = get_model()
    
    try:
        # Set a 2-minute timeout for faster generation
        with timeout_context(120):
            return model.generate_latex_review(paper_content)
    except (TimeoutError, Exception) as e:
        print(f"Generation timed out or failed: {e}")
        print("Using fallback template...")
        return model._generate_fallback_review()