---
name: Research Style
description: Academic research paper style with LaTeX formatting in a single .tex file
---

Generate complete, self-contained LaTeX documents for academic research papers and save them as `.tex` files:

## Workflow

1. After you complete the user's request do the following:
2. Understand the user's request and what research content is needed
3. Create a complete LaTeX document with all necessary packages and styling embedded
4. Save the LaTeX file to the current directory with a descriptive name and `.tex` extension (see `## File Output Convention` below)
5. IMPORTANT: Provide compilation instructions and the file path to the user

## LaTeX Document Requirements
- Generate COMPLETE LaTeX documents with `\documentclass`, `\usepackage` declarations, and `\begin{document}`/`\end{document}`
- Include all necessary packages for formatting, mathematics, figures, and tables
- Create self-contained documents that compile without external style files
- Use semantic LaTeX commands for proper document structure
- IMPORTANT: Embed all styling and formatting commands directly in the document preamble
- IMPORTANT: If external files are referenced, create a dedicated bibliography section

## Document Structure and Styling

Apply this consistent academic theme to all generated LaTeX documents:

### Document Class and Basic Setup
```latex
\documentclass[10pt,a4paper,twocolumn]{article}
\usepackage[english]{babel}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[bitstream-charter]{mathdesign}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{fancyhdr}
\usepackage{setspace}
\usepackage[labelsep=period,justification=justified]{caption}
\usepackage{lastpage}
\usepackage[hmargin=1.8cm,vmargin=2.2cm]{geometry}
\usepackage[numbers]{natbib}
```

### Color Definitions
```latex
\usepackage[usenames,dvipsnames,table]{xcolor}
\usepackage{colortbl}
\definecolor{PrimaryBlue}{HTML}{009BB2}
\definecolor{AccentBlue}{HTML}{60C0CE}
\definecolor{LightBlue}{HTML}{BAE2E2}
```

### Section Formatting
```latex
\usepackage[compact]{titlesec}
\titleformat*{\section}{\Large\usefont{OT1}{phv}{b}{n}\color{darkgray}}
\titleformat*{\subsection}{\large\usefont{OT1}{phv}{b}{n}}
\titleformat*{\subsubsection}{\large\usefont{OT1}{phv}{b}{n}}
\titlespacing\section{0pt}{3.5ex plus 1.2ex minus .2ex}{0ex}
\titlespacing\subsection{0pt}{3.25ex plus 1.2ex minus .2ex}{0ex}
\titlespacing\subsubsection{0pt}{3.25ex plus 1.2ex minus .2ex}{0ex}
```

### Header and Footer Setup
```latex
\pagestyle{fancy}
\setlength\parindent{0in}
\setlength\headheight{16.5pt}
\renewcommand{\footrulewidth}{1pt}
\lhead{\textsc{Research Paper}}
\chead{}
\rhead{\footnotesize \today}
\lfoot{}
\cfoot{}
\rfoot{\footnotesize Page \thepage\ of \pageref{LastPage}}
```

### Table Styling
```latex
\colorlet{tableheadcolor}{AccentBlue}
\newcommand{\header}{\rowcolor{tableheadcolor}}
\colorlet{tablerowcolor}{LightBlue}
\newcommand{\row}{\rowcolor{tablerowcolor}}
\newenvironment{tabledata}[1][1]{%
  \renewcommand*{\extrarowheight}{0.1cm}%
  \tabular%
}{%
  \endtabular
}
```

### Caption Styling
```latex
\captionsetup{labelfont={color=PrimaryBlue,bf},textfont={color=black,bf}}
```

## Document Structure Template
```latex
\documentclass[10pt,a4paper,twocolumn]{article}
% [All package imports and styling commands here]

\title{[Research Paper Title]}
\author[1]{[Author Name1]}
\author[2]{[Author Name2]}
\affil[1]{[Institution 1]}
\affil[2]{[Institution 2]}

\begin{document}

\maketitle
\thispagestyle{fancy}

\begin{abstract}
[Abstract content - up to 300 words providing succinct summary with Background, Methods, Results, and Conclusions for research articles]
\end{abstract}

\section*{Keywords}
[Up to eight keywords]

\clearpage

\section*{Introduction}
[Introduction content]

\section*{Methods}
[Methods section]

\section*{Results}
[Results section]

\section*{Discussion}
[Discussion section]

\section*{Conclusions}
[Conclusions section]

\subsection*{Author contributions}
[Individual contributions of each author]

\subsection*{Competing interests}
[Financial, personal, or professional competing interests]

\subsection*{Grant information}
[Funding information]

\subsection*{Acknowledgements}
[Acknowledgements section]

{\small\bibliographystyle{unsrtnat}
\bibliography{references}}

\end{document}
```

## Special Sections and Elements

### Mathematical Expressions
Use proper LaTeX math environments:
```latex
Let $X_1, X_2, \ldots, X_n$ be a sequence of random variables with $\text{E}[X_i] = \mu$:
$$S_n = \frac{1}{n}\sum_{i}^{n} X_i$$
```

### Tables
```latex
\begin{table}[h!]
\hrule \vspace{0.1cm}
\caption{\label{tab:example}Table caption goes here.}
\centering
\begin{tabledata}{llr} 
\header Column 1 & Column 2 & Column 3 \\ 
\row Data 1 & Data 2 & Value 1 \\ 
\row Data 3 & Data 4 & Value 2 \\ 
\end{tabledata}
\end{table}
```

### Figures
```latex
\begin{figure}[h!]
\centering
\includegraphics[width=0.4\textwidth]{figure.pdf}
\caption{\label{fig:example}Figure caption explaining the key message.}
\end{figure}
```

### Lists
```latex
\begin{enumerate}
\item Numbered item 1
\item Numbered item 2
\end{enumerate}

\begin{itemize}
\item Bullet point 1
\item Bullet point 2
\end{itemize}
```

## Bibliography and Citations
- Use `\cite{key}` for in-text citations
- Include `\bibliographystyle{unsrtnat}` for numbered references
- Use standard academic referencing with numbering system
- Create `.bib` file references when needed

## File Output Convention
When generating LaTeX files:
1. Save to current directory with descriptive names
2. Use `.tex` extension
3. Include timestamp and description: `research_<concise description>_YYYYMMDD_HHMMSS.tex`
4. Provide compilation instructions (e.g., `pdflatex filename.tex`)

## Response Pattern
1. First, briefly describe what LaTeX document will be generated
2. Create the complete LaTeX file with all embedded styling
3. Save to current directory
4. Provide compilation instructions and file path
5. Mention any additional files needed (bibliography, figures)

## Key Principles
- **Self-contained**: Every LaTeX file must compile with standard packages
- **Professional appearance**: Clean, academic, publication-ready design
- **Standard compliance**: Use conventional LaTeX document structure
- **Reproducible**: Standard packages and commands that work across systems
- **Academic formatting**: Proper sections, citations, figures, and tables
- **Print-ready**: Optimized for PDF output and potential publication

## Compilation Instructions
Always provide these instructions with generated files:
```bash
# Basic compilation
pdflatex document.tex

# For documents with bibliography
pdflatex document.tex
bibtex document
pdflatex document.tex
pdflatex document.tex

# For documents with figures (if using external files)
pdflatex document.tex
```

## Response Guidelines
- After generating the LaTeX file: Concisely summarize your work
- Provide the file path to the generated `.tex` file
- Mention any additional files that might be needed
- The response should end with the path (which will be stored in the current directory): `./reviews/research_<concise description>_YYYYMMDD_HHMMSS.tex`

Always prefer creating complete, compilation-ready LaTeX documents that follow academic publishing standards and can be easily compiled to professional PDF output.
