"""
chains/style_chains.py
LangChain chains for style prompt construction and style transfer.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from app.extractors.style_extractor import StyleFingerprint


def get_llm(api_key: str):
    """Initialize Gemini 2.5 Pro LLM."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.7,
    )


# ── Chain 1: Style Analysis Chain ────────────────────────────────────────────

STYLE_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert linguist and writing style analyst.
Your job is to deeply analyze a writer's style and produce a rich, 
nuanced description of how they write. Be specific, concrete, and insightful.
Focus on what makes this writer UNIQUE compared to others."""),
    ("human", """Here is a quantitative style fingerprint extracted from the writer's samples:

{style_description}

Raw metrics:
- Avg sentence length: {avg_sentence_length} words
- Vocabulary richness: {vocabulary_richness}
- Readability score: {readability_score}/100
- Adverb ratio: {adverb_ratio}
- Adjective ratio: {adjective_ratio}
- Passive voice ratio: {passive_voice_ratio}
- First person rate: {first_person_rate}
- Uses contractions: {uses_contractions}
- Uses hedging language: {uses_hedging}
- Tone detected: {tone}
- Common sentence starters: {common_sentence_starters}
- Signature phrases: {signature_phrases}
- Favourite words: {top_words}
- Ellipsis usage: {ellipsis_count}
- Dash usage: {dash_usage}
- Exclamation rate: {exclamation_rate}

Based on ALL of these metrics, write a detailed, human-readable style profile (200-300 words) 
that captures this writer's voice. Include:
1. Overall personality/voice impression
2. Sentence rhythm and structure tendencies  
3. Word choice and vocabulary preferences
4. Punctuation habits and what they signal
5. Any unique quirks or signature patterns

Write it as if briefing a ghostwriter who needs to perfectly mimic this person.""")
])


def build_style_analysis_chain(api_key: str):
    """Chain that produces a rich style analysis from fingerprint."""
    llm = get_llm(api_key)
    return STYLE_ANALYSIS_PROMPT | llm | StrOutputParser()


# ── Chain 2: Style Transfer Chain ────────────────────────────────────────────

STYLE_TRANSFER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a master ghostwriter specializing in voice matching.
Your task is to rewrite AI-generated text so it sounds EXACTLY like a specific human writer.
You must preserve all the original meaning, facts, and information — only the style should change.

Rules:
- Keep ALL information from the original text intact
- Match the writer's voice so closely that they could claim it as their own
- Do NOT add new information or opinions not in the original
- Do NOT use generic AI writing patterns
- Output ONLY the rewritten text, nothing else"""),
    ("human", """WRITER'S STYLE PROFILE:
{rich_style_analysis}

QUANTITATIVE STYLE REQUIREMENTS:
{style_description}

Specific patterns to replicate:
- Target sentence length: ~{avg_sentence_length} words per sentence
- Tone: {tone}
- Uses contractions: {uses_contractions}
- First person frequency: {first_person_rate}
- Signature phrases to weave in naturally: {signature_phrases}
- Punctuation habits: exclamation={exclamation_rate}, ellipsis={ellipsis_count}, dashes={dash_usage}

AI-GENERATED TEXT TO REWRITE:
{ai_text}

Rewrite the above text in the writer's exact voice:""")
])


def build_style_transfer_chain(api_key: str):
    """Chain that rewrites AI text in the user's style."""
    llm = get_llm(api_key)
    return STYLE_TRANSFER_PROMPT | llm | StrOutputParser()


# ── Chain 3: Quality Check Chain ─────────────────────────────────────────────

QUALITY_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a writing quality evaluator. 
Evaluate how well a rewritten text matches a target writing style.
Be objective and specific. Return ONLY a JSON object, no markdown."""),
    ("human", """STYLE PROFILE:
{style_description}

ORIGINAL AI TEXT:
{ai_text}

REWRITTEN TEXT:
{rewritten_text}

Evaluate the rewrite on these dimensions (score 0-10 each):
1. style_match: How well does it match the described style?
2. content_preservation: Is all original information preserved?
3. naturalness: Does it sound like a real human wrote it?
4. voice_consistency: Is the voice consistent throughout?

Return ONLY this JSON (no markdown, no backticks):
{{
  "style_match": <0-10>,
  "content_preservation": <0-10>,
  "naturalness": <0-10>,
  "voice_consistency": <0-10>,
  "overall_score": <0-10>,
  "feedback": "<one sentence of specific feedback>",
  "what_worked": "<what style elements were captured well>",
  "what_to_improve": "<what could be improved>"
}}""")
])


def build_quality_check_chain(api_key: str):
    """Chain that scores the quality of style transfer."""
    llm = get_llm(api_key)
    return QUALITY_CHECK_PROMPT | llm | StrOutputParser()
