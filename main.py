# ***********************************************************************************************************
# **********************************         TEXT REPLICA FAST API         **********************************
# ***********************************************************************************************************




#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LIBRARY IMPORT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import os
import tempfile
import shutil
from typing import List
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.agents.style_graph import run_style_transfer

from dotenv import load_dotenv
load_dotenv()

import os
import tempfile

#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< APP INIT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

app = FastAPI(
    title="TextReplica API",
    description="TextReplica — Transfer your writing style to any text.",
    version="1.0.0",
)

#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FOR HTML CONNECTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< RESPONSE MODELS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class QualityScores(BaseModel):
    overall_score:          float = 0
    style_match:            float = 0
    content_preservation:   float = 0
    naturalness:            float = 0
    voice_consistency:      float = 0
    feedback:               str = ""
    what_worked:            str = ""
    what_to_improve:        str = ""


class StyleFingerprint(BaseModel):
    avg_sentence_length:    float = 0
    readability_score:      float = 0
    vocabulary_richness:    float = 0
    tone:                   str = "N/A"
    exclamation_rate:       float = 0
    ellipsis_count:         int = 0
    dash_usage:             int = 0
    uses_parentheses:       bool = False
    comma_rate:             float = 0
    uses_contractions:      bool = False
    first_person_rate:      float = 0
    uses_hedging:           bool = False
    passive_voice_ratio:    float = 0
    adverb_ratio:           float = 0
    adjective_ratio:        float = 0
    top_words:              List[str] = []
    signature_phrases:      List[str] = []


class TransferResponse(BaseModel):
    rewritten_text: str
    fingerprint: StyleFingerprint
    rich_style_analysis: str
    quality_scores: QualityScores


#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CHECK API HEALTH >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

@app.get("/health", tags=["Utility"])
def health_check():
    return {"status": "ok"}


# ── Main Transfer Endpoint ────────────────────────────────────────────────────

@app.post("/transfer", response_model=TransferResponse, tags=["Style Transfer"])
async def transfer_style(
    ai_text: str = Form(..., description="The AI-generated text to be rewritten"),
    files: List[UploadFile] = File(..., description="Your writing samples (.txt, .pdf, .md)"),
):
    """
    Upload writing samples and AI-generated text.
    Returns the rewritten text, style fingerprint, and quality report.
    """

    # ── Validate inputs ───────────────────────────────────────────────────────
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured on server.")
    
    if not ai_text.strip():
        raise HTTPException(status_code=400, detail="AI-generated text is required.")

    if not files:
        raise HTTPException(status_code=400, detail="At least one writing sample file is required.")

    allowed_extensions = {".txt", ".pdf", ".md"}
    for f in files:
        ext = os.path.splitext(f.filename)[-1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}' for file '{f.filename}'. Allowed: .txt, .pdf, .md",
            )

#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< SAVE UPLOADED FILE TO TEMP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    temp_dir = tempfile.mkdtemp()
    file_paths = []

    try:
        for uf in files:
            dest_path = os.path.join(temp_dir, uf.filename)
            with open(dest_path, "wb") as out_file:
                content = await uf.read()
                out_file.write(content)
            file_paths.append(dest_path)

        # STYLE TRANSFER PIPELINE

        result: dict = run_style_transfer(
            file_paths=file_paths,
            ai_text=ai_text,
            api_key=api_key,
        )

    finally:
        # Clean temp file
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Handle Error

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< BUILD RESPONSE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    raw_fingerprint = result.get("fingerprint", {})
    raw_quality_score = result.get("quality_scores", {})

    fingerprint = StyleFingerprint(
        avg_sentence_length =   raw_fingerprint.get("avg_sentence_length", 0),
        readability_score   =   raw_fingerprint.get("readability_score", 0),
        vocabulary_richness =   raw_fingerprint.get("vocabulary_richness", 0),
        tone                =   raw_fingerprint.get("tone", "N/A"),
        exclamation_rate    =   raw_fingerprint.get("exclamation_rate", 0),
        ellipsis_count      =   raw_fingerprint.get("ellipsis_count", 0),
        dash_usage          =   raw_fingerprint.get("dash_usage", 0),
        uses_parentheses    =   raw_fingerprint.get("uses_parentheses", False),
        comma_rate          =   raw_fingerprint.get("comma_rate", 0),
        uses_contractions   =   raw_fingerprint.get("uses_contractions", False),
        first_person_rate   =   raw_fingerprint.get("first_person_rate", 0),
        uses_hedging        =   raw_fingerprint.get("uses_hedging", False),
        passive_voice_ratio =   raw_fingerprint.get("passive_voice_ratio", 0),
        adverb_ratio        =   raw_fingerprint.get("adverb_ratio", 0),
        adjective_ratio     =   raw_fingerprint.get("adjective_ratio", 0),
        top_words           =   raw_fingerprint.get("top_words", []),
        signature_phrases   =   raw_fingerprint.get("signature_phrases", []),
    )

    quality_scores = QualityScores(
        overall_score       =   raw_quality_score.get("overall_score", 0),
        style_match         =   raw_quality_score.get("style_match", 0),
        content_preservation=   raw_quality_score.get("content_preservation", 0),
        naturalness         =   raw_quality_score.get("naturalness", 0),
        voice_consistency   =   raw_quality_score.get("voice_consistency", 0),
        feedback            =   raw_quality_score.get("feedback", ""),
        what_worked         =   raw_quality_score.get("what_worked", ""),
        what_to_improve     =   raw_quality_score.get("what_to_improve", ""),
    )



#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< RETURN RESPONSE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    return TransferResponse(
        rewritten_text      =   result.get("rewritten_text", ""),
        fingerprint         =   fingerprint,
        rich_style_analysis =   result.get("rich_style_analysis", ""),
        quality_scores      =   quality_scores,
    )

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
