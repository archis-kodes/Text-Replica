"""
agents/style_graph.py
LangGraph orchestration of the full style transfer pipeline.

Graph flow:
  parse_files → extract_style → analyze_style → transfer_style → quality_check → END
"""

import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from app.extractors.style_extractor import extract_style_fingerprint, StyleFingerprint
from app.chains.style_chains import (
    build_style_analysis_chain,
    build_style_transfer_chain,
    build_quality_check_chain,
)
from app.utils.file_parser import parse_multiple_files


# ── State Definition ──────────────────────────────────────────────────────────

class StyleTransferState(TypedDict):
    # Inputs
    file_paths: list[str]
    ai_text: str
    api_key: str

    # Intermediate outputs
    combined_sample_text: str
    fingerprint: dict          # StyleFingerprint as dict
    rich_style_analysis: str   # LLM-generated style description
    rewritten_text: str        # Final style-transferred output

    # Quality evaluation
    quality_scores: dict

    # Error tracking
    error: str
    current_step: str


# ── Node Functions ────────────────────────────────────────────────────────────

def node_parse_files(state: StyleTransferState) -> dict:
    """Node 1: Parse uploaded files into combined text."""
    try:
        combined = parse_multiple_files(state["file_paths"])
        if not combined.strip():
            return {"error": "No text could be extracted from the uploaded files.", "current_step": "parse_files"}
        return {
            "combined_sample_text": combined,
            "current_step": "parse_files",
            "error": ""
        }
    except Exception as e:
        return {"error": f"File parsing failed: {str(e)}", "current_step": "parse_files"}


def node_extract_style(state: StyleTransferState) -> dict:
    """Node 2: Extract quantitative style fingerprint using spaCy + textstat."""
    try:
        text = state["combined_sample_text"]
        if len(text.split()) < 50:
            return {
                "error": "Writing samples are too short. Please provide at least 50 words of sample text.",
                "current_step": "extract_style"
            }
        fingerprint: StyleFingerprint = extract_style_fingerprint(text)
        return {
            "fingerprint": fingerprint.model_dump(),
            "current_step": "extract_style",
            "error": ""
        }
    except Exception as e:
        return {"error": f"Style extraction failed: {str(e)}", "current_step": "extract_style"}


def node_analyze_style(state: StyleTransferState) -> dict:
    """Node 3: Use Gemini to produce a rich, human-readable style analysis."""
    try:
        fp = state["fingerprint"]
        chain = build_style_analysis_chain(state["api_key"])
        rich_analysis = chain.invoke({
            "style_description": fp["style_description"],
            "avg_sentence_length": fp["avg_sentence_length"],
            "vocabulary_richness": fp["vocabulary_richness"],
            "readability_score": fp["readability_score"],
            "adverb_ratio": fp["adverb_ratio"],
            "adjective_ratio": fp["adjective_ratio"],
            "passive_voice_ratio": fp["passive_voice_ratio"],
            "first_person_rate": fp["first_person_rate"],
            "uses_contractions": fp["uses_contractions"],
            "uses_hedging": fp["uses_hedging"],
            "tone": fp["tone"],
            "common_sentence_starters": fp["common_sentence_starters"],
            "signature_phrases": fp["signature_phrases"],
            "top_words": fp["top_words"],
            "ellipsis_count": fp["ellipsis_count"],
            "dash_usage": fp["dash_usage"],
            "exclamation_rate": fp["exclamation_rate"],
        })
        return {
            "rich_style_analysis": rich_analysis,
            "current_step": "analyze_style",
            "error": ""
        }
    except Exception as e:
        return {"error": f"Style analysis failed: {str(e)}", "current_step": "analyze_style"}


def node_transfer_style(state: StyleTransferState) -> dict:
    """Node 4: Rewrite the AI-generated text in the user's style."""
    try:
        fp = state["fingerprint"]
        chain = build_style_transfer_chain(state["api_key"])
        rewritten = chain.invoke({
            "rich_style_analysis": state["rich_style_analysis"],
            "style_description": fp["style_description"],
            "avg_sentence_length": fp["avg_sentence_length"],
            "tone": fp["tone"],
            "uses_contractions": fp["uses_contractions"],
            "first_person_rate": fp["first_person_rate"],
            "signature_phrases": ", ".join(fp["signature_phrases"][:5]) if fp["signature_phrases"] else "none detected",
            "exclamation_rate": fp["exclamation_rate"],
            "ellipsis_count": fp["ellipsis_count"],
            "dash_usage": fp["dash_usage"],
            "ai_text": state["ai_text"],
        })
        return {
            "rewritten_text": rewritten,
            "current_step": "transfer_style",
            "error": ""
        }
    except Exception as e:
        return {"error": f"Style transfer failed: {str(e)}", "current_step": "transfer_style"}


def node_quality_check(state: StyleTransferState) -> dict:
    """Node 5: Score the quality of the style transfer."""
    try:
        fp = state["fingerprint"]
        chain = build_quality_check_chain(state["api_key"])
        result = chain.invoke({
            "style_description": fp["style_description"],
            "ai_text": state["ai_text"],
            "rewritten_text": state["rewritten_text"],
        })

        # Clean and parse JSON
        clean = result.strip().replace("```json", "").replace("```", "").strip()
        scores = json.loads(clean)
        return {
            "quality_scores": scores,
            "current_step": "quality_check",
            "error": ""
        }
    except Exception as e:
        # Non-fatal: return dummy scores if parsing fails
        return {
            "quality_scores": {
                "style_match": 0,
                "content_preservation": 0,
                "naturalness": 0,
                "voice_consistency": 0,
                "overall_score": 0,
                "feedback": "Quality check could not be completed.",
                "what_worked": "N/A",
                "what_to_improve": "N/A",
            },
            "current_step": "quality_check",
            "error": ""
        }


# ── Edge Conditions ───────────────────────────────────────────────────────────

def should_continue(state: StyleTransferState) -> str:
    """Stop the graph early if any node produced an error."""
    if state.get("error"):
        return END
    return "continue"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_style_transfer_graph():
    """Build and compile the LangGraph pipeline."""

    graph = StateGraph(StyleTransferState)

    # Register nodes
    graph.add_node("parse_files", node_parse_files)
    graph.add_node("extract_style", node_extract_style)
    graph.add_node("analyze_style", node_analyze_style)
    graph.add_node("transfer_style", node_transfer_style)
    graph.add_node("quality_check", node_quality_check)

    # Entry point
    graph.set_entry_point("parse_files")

    # Conditional edges — stop on error, else continue
    graph.add_conditional_edges("parse_files", should_continue, {
        "continue": "extract_style",
        END: END
    })
    graph.add_conditional_edges("extract_style", should_continue, {
        "continue": "analyze_style",
        END: END
    })
    graph.add_conditional_edges("analyze_style", should_continue, {
        "continue": "transfer_style",
        END: END
    })
    graph.add_conditional_edges("transfer_style", should_continue, {
        "continue": "quality_check",
        END: END
    })
    graph.add_edge("quality_check", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_style_transfer(
    file_paths: list[str],
    ai_text: str,
    api_key: str,
) -> StyleTransferState:
    """
    Run the full style transfer pipeline.

    Args:
        file_paths: List of paths to user writing sample files
        ai_text: The AI-generated text to rewrite
        api_key: Google Gemini API key

    Returns:
        Final pipeline state with all intermediate results
    """
    graph = build_style_transfer_graph()

    initial_state: StyleTransferState = {
        "file_paths": file_paths,
        "ai_text": ai_text,
        "api_key": api_key,
        "combined_sample_text": "",
        "fingerprint": {},
        "rich_style_analysis": "",
        "rewritten_text": "",
        "quality_scores": {},
        "error": "",
        "current_step": "start",
    }

    final_state = graph.invoke(initial_state)
    return final_state
