# ✍️ VoiceClone — Personal Style Transfer System

Make AI-generated content sound exactly like **you** wrote it.

Built with **LangChain** · **LangGraph** · **Gemini 2.5 Pro** · **FastAPI**

---

## 🧠 How It Works

```
Your Writing Samples (.txt / .pdf / .md)
            ↓
    📄 File Parser (PyMuPDF, plain text)
            ↓
    🔬 Style Extractor (spaCy + textstat)
       → sentence length, vocabulary richness
       → punctuation habits, tone, voice traits
       → signature phrases, favourite words
            ↓
    🧠 Gemini Style Analyst
       → Rich human-readable voice profile
            ↓
    ✍️  Gemini Style Transfer
       → AI text rewritten in YOUR voice
            ↓
    📊 Quality Evaluator
       → Scores: style match, naturalness, content preservation
```

---

## 🗂️ Project Structure

```
style_transfer/
│
├── main.py             # Main UI
│
├── app/
│   ├── agents/
│   │   └── style_graph.py       # LangGraph pipeline orchestration
│   │
│   ├── chains/
│   │   └── style_chains.py      # LangChain prompts & chains (Gemini)
│   │
│   ├── extractors/
│   │   └── style_extractor.py   # NLP style fingerprint extraction
│   │
│   └── utils/
│       └── file_parser.py       # File parsing (.txt, .pdf, .md)
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Setup & Run

### 1. Clone / unzip the project

```bash
cd style_transfer
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 5. Set up your API key

```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key
```

Get your free Gemini API key at: https://aistudio.google.com

### 6. Run the app

```bash
uvicorn main:app --reload
```

---

## 📐 Style Features Extracted

| Category | What's measured |
|---|---|
| **Surface** | Sentence length, word length, vocabulary richness, readability |
| **Punctuation** | Exclamation marks, ellipses, dashes, commas, parentheses |
| **Grammar** | Adverb/adjective/verb ratios, passive vs active voice |
| **Vocabulary** | Top words, signature phrases, contractions, first-person rate |
| **Tone** | Casual / semi-formal / formal detection |

---

## 🧩 LangGraph Pipeline Nodes

| Node | Description |
|---|---|
| `parse_files` | Reads uploaded files and extracts text |
| `extract_style` | Runs NLP analysis to build style fingerprint |
| `analyze_style` | Gemini produces a rich style narrative |
| `transfer_style` | Gemini rewrites AI text in user's voice |
| `quality_check` | Gemini scores the quality of the transfer |

---

## 💡 Tips for Best Results

- Upload **5–10 writing samples** for a richer style fingerprint
- Samples should be **at least 200 words each**
- Use writing that reflects your **natural voice** (emails, blogs, essays)
- The more varied the samples, the better the style capture

---

## 🛠️ Tech Stack

- **LangGraph** — Pipeline orchestration with stateful nodes
- **LangChain** — Prompt chaining and LLM abstraction
- **Gemini 2.5 Pro** — Style analysis and transfer
- **spaCy** — NLP tokenization, POS tagging, dependency parsing
- **textstat** — Readability scoring
- **PyMuPDF** — PDF text extraction
- **FastAPI** — API
- **Pydantic** — Data validation for style fingerprint

---

## 📊 Quality Scoring

After every transfer, the system evaluates:

- **Style Match** — Does the output match your style profile?
- **Content Preservation** — Is all original information intact?
- **Naturalness** — Does it sound like a real human wrote it?
- **Voice Consistency** — Is the voice consistent throughout?

Each scored 0–10. Results can be downloaded as JSON.