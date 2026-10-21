# LLM Hallucination Detection and Correction Framework

A local Flask web application that decomposes AI-generated text into factual claims, retrieves relevant evidence in parallel batches, validates each claim against evidence using a DeBERTa NLI cross-encoder, and generates validated corrections using local Ollama.

## Architecture & Capabilities

- **Claim Decomposition:** Splits paragraphs into individual claims while preserving abbreviations (e.g., `Dr.`, `U.S.`, `sq. km.`), numeric decimals (`3.14`), and bullet lists.
- **Parallel Batch Evidence Retrieval:** Queries both a local JSON knowledge base (exact and fuzzy match) and the Wikipedia public API. Searches and page extracts are batched in a single network request using connection pooling.
- **Passage-Focused NLI Verification:** Uses `cross-encoder/nli-deberta-v3-small` with sentence-level passage extraction to keep evidence well within the 512-token limit, avoiding attention dilution.
- **Dynamic Label Mapping:** Correctly maps model classification logits (`contradiction` $\rightarrow$ Contradicted, `entailment` $\rightarrow$ Supported, `neutral` $\rightarrow$ Unverified) dynamically from the model configuration.
- **NLI-Grounded Correction:** For contradicted statements, proposes factual corrections using local Ollama (`llama3.2:1b`) and validates them using the DeBERTa NLI model before substitution.
- **Non-Destructive Paragraph Rebuilding:** Preserves document coherence without deleting unverified sentences.
- **Multi-Threaded Execution:** Analyzes sentences concurrently via a thread pool to provide fast response times.
- **Analysis History:** Maintains a rolling history of recent session analyses accessible via the web interface.

## Requirements

- Python 3.10 or 3.11 recommended (PyTorch and transformers have stable binary wheels for 3.10 and 3.11).
- Internet connection for Wikipedia evidence retrieval and initial model download.
- (Optional) [Ollama](https://ollama.com) running `llama3.2:1b` for automatic correction generation.

## Windows PowerShell Setup

```powershell
cd path\to\LLM_Hallucination_Framework
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-ml.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your web browser.

If PowerShell blocks execution scripts:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Or run via the venv python interpreter directly:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe app.py
```

## Optional Local Ollama Setup (Corrections)

1. Download and install Ollama from [https://ollama.com](https://ollama.com).
2. Pull the lightweight correction model:
   ```powershell
   ollama pull llama3.2:1b
   ```
3. Keep Ollama running. If Ollama is offline or uninstalled, claim verification continues functioning normally with safe fallbacks.

## Starter Verification Examples

```text
The Earth revolves around the Sun.
The Earth is the largest planet in the Solar System.
Telangana became a state in India in 2014.
The Moon is made entirely of cheese.
```

## Project Layout

```text
LLM_Hallucination_Framework/
  app.py                   # Flask server, concurrent sentence processing, API routes
  requirements.txt         # Core web and HTTP requirements
  requirements-ml.txt      # Machine learning and NLI cross-encoder requirements
  knowledge_base.json      # Local curated reference facts
  services/
    __init__.py
    evidence.py            # Batched Wikipedia API and local knowledge-base retrieval
    verification.py        # DeBERTa NLI inference and passage extraction
    correction.py          # Ollama correction generation and NLI validation
  templates/
    index.html             # Web dashboard
  static/
    style.css              # Dashboard styling and responsive layout
    script.js              # Client-side UI logic, history tracking, and async API calls
```

## API Endpoints

- `GET /api/health` — Checks backend availability and verification model load status.
- `POST /api/analyze` — Analyzes a paragraph: `{"text": "Your text..."}`. Returns claim status, explanations, evidence citations, and corrected text.
- `POST /api/chat` — Factual question / single-statement verification query.
- `GET /api/history` — Returns recent session analysis summaries.
- `POST /api/history/clear` — Clears stored session analysis history.

