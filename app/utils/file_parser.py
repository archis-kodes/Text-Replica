"""
utils/file_parser.py
Handles parsing of .txt, .pdf, and .md files into plain text.
"""

import fitz  # PyMuPDF
from pathlib import Path


def parse_file(file_path: str) -> str:
    """
    Parse a file and return its text content.
    Supports: .txt, .md, .pdf
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in [".txt", ".md"]:
        return _parse_text(path)
    elif suffix == ".pdf":
        return _parse_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .txt, .md, .pdf")


def _parse_text(path: Path) -> str:
    """Read plain text or markdown files."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1").strip()


def _parse_pdf(path: Path) -> str:
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(str(path))
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts).strip()


def parse_multiple_files(file_paths: list[str]) -> str:
    """
    Parse multiple files and combine their text.
    Returns combined text separated by newlines.
    """
    all_text = []
    errors = []

    for file_path in file_paths:
        try:
            text = parse_file(file_path)
            if text:
                all_text.append(text)
        except Exception as e:
            errors.append(f"{file_path}: {str(e)}")

    if errors:
        print(f"[FileParser] Warnings: {errors}")

    return "\n\n---\n\n".join(all_text)
