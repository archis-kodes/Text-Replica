"""
extractors/style_extractor.py
Extracts a detailed style fingerprint from user writing samples.
Uses spaCy + textstat + custom heuristics.
"""


#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LIBRARY IMPORT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


import re
import spacy
import textstat
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from pydantic import BaseModel

# Download Tokenizers
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Download Stopwords
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


class StyleFingerprint(BaseModel):
    """Structured style fingerprint extracted from user writing."""

    # Surface stats
    avg_sentence_length:    float
    avg_word_length:        float
    vocabulary_richness:    float
    readability_score:      float
    avg_syllables_per_word: float

    # Punctuation habits
    exclamation_rate:       float
    question_rate:          float
    ellipsis_count:         int
    dash_usage:             int
    comma_rate:             float
    uses_parentheses:       bool
    avg_paragraph_length:   float

    # Grammar & syntax
    adverb_ratio:           float
    adjective_ratio:        float
    verb_ratio:             float
    noun_ratio:             float
    passive_voice_ratio:    float
    common_sentence_starters: list[str]

    # Vocabulary & personality
    top_words:              list[str]
    signature_phrases:      list[str]
    uses_contractions:      bool
    first_person_rate:      float
    uses_hedging:           bool
    tone:                   str  # casual / formal / semi-formal

    # Human-readable summary
    style_description: str


def extract_style_fingerprint(text: str) -> StyleFingerprint:
    """Main function: extract full style fingerprint from combined writing samples."""

    doc = nlp(text[:500000])  # Cap to avoid memory issues
    sentences = list(doc.sents)
    words = [t for t in doc if t.is_alpha]
    word_texts = [w.text.lower() for w in words]

    surface =   _surface_stats(text, sentences, words)
    punct =     _punctuation_style(text)
    grammar =   _grammar_style(doc, words, sentences)
    vocab =     _vocabulary_style(doc, text, word_texts)
    tone =      _detect_tone(surface["readability_score"], vocab["first_person_rate"])

    description = _build_style_description(surface, punct, grammar, vocab, tone)

    return StyleFingerprint(
        # Surface
        avg_sentence_length     =   surface["avg_sentence_length"],
        avg_word_length         =   surface["avg_word_length"],
        vocabulary_richness     =   surface["vocabulary_richness"],
        readability_score       =   surface["readability_score"],
        avg_syllables_per_word  =   surface["avg_syllables_per_word"],
        # Punctuation
        exclamation_rate        =   punct["exclamation_rate"],
        question_rate           =   punct["question_rate"],
        ellipsis_count          =   punct["ellipsis_count"],
        dash_usage              =   punct["dash_usage"],
        comma_rate              =   punct["comma_rate"],
        uses_parentheses        =   punct["uses_parentheses"],
        avg_paragraph_length    =   punct["avg_paragraph_length"],
        # Grammar
        adverb_ratio            =   grammar["adverb_ratio"],
        adjective_ratio         =   grammar["adjective_ratio"],
        verb_ratio              =   grammar["verb_ratio"],
        noun_ratio              =   grammar["noun_ratio"],
        passive_voice_ratio     =   grammar["passive_voice_ratio"],
        common_sentence_starters=   grammar["common_sentence_starters"],
        # Vocabulary
        top_words=vocab["top_words"],
        signature_phrases       =   vocab["signature_phrases"],
        uses_contractions       =   vocab["uses_contractions"],
        first_person_rate       =   vocab["first_person_rate"],
        uses_hedging            =   vocab["uses_hedging"],
        tone                    =   tone,
        style_description       =   description,
    )


#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< HELPER FUNCTIONS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< SURFACE STATS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def _surface_stats(text: str, sentences, words) -> dict:
    word_count = len(words) or 1
    sent_count = len(sentences) or 1
    return {
        "avg_sentence_length": round(word_count / sent_count, 2),
        "avg_word_length": round(sum(len(w.text) for w in words) / word_count, 2),
        "vocabulary_richness": round(len(set(w.text.lower() for w in words)) / word_count, 3),
        "readability_score": round(textstat.flesch_reading_ease(text), 2),
        "avg_syllables_per_word": round(textstat.avg_syllables_per_word(text), 2),
    }


#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< PUNCTUATION STYLE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def _punctuation_style(text: str) -> dict:
    words = text.split()
    word_count = len(words) or 1
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    para_lengths = [len(p.split()) for p in paragraphs] if paragraphs else [0]

    return {
        "exclamation_rate":     round(text.count("!") / word_count, 4),
        "question_rate":        round(text.count("?") / word_count, 4),
        "ellipsis_count":       text.count("..."),
        "dash_usage":           text.count("—") + text.count(" - "),
        "comma_rate":           round(text.count(",") / word_count, 4),
        "uses_parentheses":     "(" in text,
        "avg_paragraph_length": round(sum(para_lengths) / len(para_lengths), 2),
    }


#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< GRAMMER STYLE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def _grammar_style(doc, words, sentences) -> dict:
    total = len(words) or 1
    pos_counts: dict[str, int] = {}
    for token in words:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

    passive = sum(1 for token in doc if token.dep_ == "auxpass")
    sent_count = len(sentences) or 1

    starters: list[str] = []
    for sent in sentences:
        tokens = [t for t in sent if t.is_alpha]
        if tokens:
            starters.append(tokens[0].text.lower())
    starter_counts = Counter(starters).most_common(5)

    return {
        "adverb_ratio":             round(pos_counts.get("ADV", 0) / total, 4),
        "adjective_ratio":          round(pos_counts.get("ADJ", 0) / total, 4),
        "verb_ratio":               round(pos_counts.get("VERB", 0) / total, 4),
        "noun_ratio":               round(pos_counts.get("NOUN", 0) / total, 4),
        "passive_voice_ratio":      round(passive / sent_count, 4),
        "common_sentence_starters": [w for w, _ in starter_counts],
    }


#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< VOCABULARY STYLE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def _vocabulary_style(doc, text: str, word_texts: list[str]) -> dict:
    stop_words = set(nlp.Defaults.stop_words)
    content_words = [w for w in word_texts if w not in stop_words and len(w) > 3]
    word_freq = Counter(content_words)
    top_words = [w for w, _ in word_freq.most_common(10)]

    # Signature bigrams/trigrams via TF-IDF
    signature_phrases: list[str] = []
    try:
        if len(text.split()) > 30:
            vectorizer = TfidfVectorizer(ngram_range=(2, 3), max_features=10, stop_words="english")
            vectorizer.fit([text])
            signature_phrases = vectorizer.get_feature_names_out().tolist()
    except Exception:
        pass

    contractions = bool(re.search(r"\b(I'm|don't|can't|it's|I've|won't|isn't|they're|we're|you're)\b", text, re.I))
    first_person = sum(1 for t in doc if t.text.lower() in {"i", "me", "my", "mine", "myself"})
    first_person_rate = round(first_person / (len(list(doc)) or 1), 4)

    hedging_words = {"maybe", "perhaps", "possibly", "probably", "might", "i think", "i believe", "seems", "appear"}
    uses_hedging = any(h in text.lower() for h in hedging_words)

    return {
        "top_words":            top_words,
        "signature_phrases":    signature_phrases,
        "uses_contractions":    contractions,
        "first_person_rate":    first_person_rate,
        "uses_hedging":         uses_hedging,
    }


#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TONE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def _detect_tone(readability: float, first_person_rate: float) -> str:
    if readability > 65 and first_person_rate > 0.03:
        return "casual"
    elif readability < 40:
        return "formal"
    else:
        return "semi-formal"


#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< BUILD TOTAL STYLE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def _build_style_description(surface, punct, grammar, vocab, tone) -> str:
    lines = []
    lines.append(f"Tone: {tone.upper()}")
    lines.append(f"Sentence length: ~{surface['avg_sentence_length']:.0f} words/sentence")
    lines.append(f"Readability: {surface['readability_score']:.1f}/100 (Flesch score)")
    lines.append(f"Vocabulary richness: {surface['vocabulary_richness']:.2f} (0=repetitive, 1=varied)")


#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< PUNCTUATIONS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    if punct["exclamation_rate"] > 0.05:
        lines.append("Frequently uses exclamation marks (energetic, enthusiastic)")

    if punct["ellipsis_count"] > 2:
        lines.append("Likes ellipses (...) for trailing thoughts")

    if punct["dash_usage"] > 2:
        lines.append("Uses dashes for asides and emphasis")

    if punct["uses_parentheses"]:
        lines.append("Adds parenthetical remarks")

#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< GRAMMER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    if grammar["adverb_ratio"] > 0.06:
        lines.append("Adverb-heavy writing (very, really, quickly...)")

    if grammar["adjective_ratio"] > 0.08:
        lines.append("Descriptive and expressive (high adjective use)")

    if grammar["passive_voice_ratio"] > 0.1:
        lines.append("Tends toward passive voice")

    else:
        lines.append("Prefers active voice")

#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< VOCABS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    if vocab["uses_contractions"]:
        lines.append("Uses contractions naturally (it's, don't, I'm)")
    if vocab["first_person_rate"] > 0.04:
        lines.append("Writes in first person frequently")
    if vocab["uses_hedging"]:
        lines.append("Uses hedging language (maybe, I think, perhaps)")

    if vocab["signature_phrases"]:
        lines.append(f"Signature phrases: {', '.join(vocab['signature_phrases'][:4])}")
    if vocab["top_words"]:
        lines.append(f"Favourite words: {', '.join(vocab['top_words'][:6])}")

    return "\n".join(lines)
