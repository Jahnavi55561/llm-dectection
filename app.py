from flask import Flask, render_template, request, jsonify
from datetime import datetime, timezone
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import re

from services.evidence import retrieve_evidence
from services.verification import verify_claim, model_status
from services.correction import suggest_correction, validate_correction, build_corrected_paragraph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app")

app = Flask(__name__)

# Maintain recent analysis history (last 20 analyses in memory)
RECENT_HISTORY = deque(maxlen=20)

ABBREVIATIONS = {
    'dr', 'mr', 'mrs', 'ms', 'prof', 'sr', 'jr', 'vs', 'etc', 'e.g', 'i.e', 'u.s', 'sq', 'km',
    'approx', 'fig', 'al', 'inc', 'ltd', 'corp', 'dept', 'no', 'univ', 'est', 'st',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
}

def split_sentences(text: str) -> list:
    text = (text or "").strip()
    if not text:
        return []

    # Protect decimals like 3.14 or numbers like $10.50
    text = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL>\2', text)
    # Protect common abbreviations using case-insensitive escaped matching
    for abbr in sorted(ABBREVIATIONS, key=len, reverse=True):
        text = re.sub(rf'(?i)\b{re.escape(abbr)}\.', r'\g<0><DOT>', text)
        text = text.replace('.<DOT>', '<DOT>')
    # Protect numbered list prefixes at start of line OR directly following a sentence boundary
    text = re.sub(r'(?m)^(\d+)\.\s+', r'\1<NUMDOT> ', text)
    text = re.sub(r'(?<=[.!?])\s+(\d{1,3})\.\s+', r' \1<NUMDOT> ', text)

    # Split on sentence boundaries: newlines or [.!?] followed by whitespace
    raw_splits = re.split(r'(?:[\r\n]+|(?<=[.!?])\s+)', text)
    sentences = []
    for s in raw_splits:
        s = s.replace('<DECIMAL>', '.').replace('<DOT>', '.').replace('<NUMDOT>', '.')
        s = re.sub(r'\s+', ' ', s).strip()
        if s:
            sentences.append(s)
    return sentences

def _process_single_sentence(idx: int, sentence: str) -> dict:
    try:
        evidence_pack = retrieve_evidence(sentence)
        verification = verify_claim(sentence, evidence_pack)
        status = verification.get("status", "Unverified")
        correction = ""
        correction_validated = False

        if status == "Contradicted":
            correction = suggest_correction(sentence, evidence_pack)
            if correction:
                correction_validated = validate_correction(sentence, correction, evidence_pack)

        return {
            "id": idx + 1,
            "sentence": sentence,
            "status": status,
            "confidence": verification.get("confidence", 0.0),
            "explanation": verification.get("explanation", ""),
            "evidence": evidence_pack.get("evidence", ""),
            "source": evidence_pack.get("source", ""),
            "source_url": evidence_pack.get("source_url", ""),
            "correction": correction if correction_validated else "",
            "correction_validated": bool(correction_validated)
        }
    except Exception as exc:
        logger.exception("Analysis failed for sentence: %s. Error: %s", sentence, exc)
        return {
            "id": idx + 1,
            "sentence": sentence,
            "status": "Unverified",
            "confidence": 0.0,
            "explanation": "Verification failed safely; no reliable conclusion was produced.",
            "evidence": "",
            "source": "",
            "source_url": "",
            "correction": "",
            "correction_validated": False
        }

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "service": "LLM Hallucination Detection and Correction Framework",
        "verification_model": model_status(),
        "time": datetime.now(timezone.utc).isoformat()
    })

@app.post("/api/analyze")
def analyze():
    data = request.get_json(silent=True) or request.form or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please paste text to analyze."}), 400
    if len(text) > 10000:
        return jsonify({"error": "Please keep input under 10,000 characters to prevent excessive latency."}), 413

    sentences = split_sentences(text)
    if not sentences:
        return jsonify({"error": "No valid sentences found in input."}), 400
    if len(sentences) > 30:
        return jsonify({"error": "Maximum 30 sentences per analysis allowed."}), 400

    # Process sentences concurrently using a thread pool for fast parallel retrieval & inference
    results = [None] * len(sentences)
    max_workers = min(6, len(sentences))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_process_single_sentence, i, sentence): i
            for i, sentence in enumerate(sentences)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error("Unexpected worker error on sentence %d: %s", idx, exc)
                results[idx] = {
                    "id": idx + 1,
                    "sentence": sentences[idx],
                    "status": "Unverified",
                    "explanation": "Worker execution error.",
                    "evidence": "",
                    "source": "",
                    "source_url": "",
                    "correction": "",
                    "correction_validated": False
                }

    counts = {
        "Supported": sum(1 for x in results if x and x["status"] == "Supported"),
        "Contradicted": sum(1 for x in results if x and x["status"] == "Contradicted"),
        "Unverified": sum(1 for x in results if x and x["status"] == "Unverified")
    }
    corrected_paragraph = build_corrected_paragraph(results)

    response_payload = {
        "results": results,
        "analysis": results,
        "counts": counts,
        "summary": counts,
        "corrected_paragraph": corrected_paragraph,
        "total_sentences": len(results),
        "supported_count": counts["Supported"],
        "contradicted_count": counts["Contradicted"],
        "unverified_count": counts["Unverified"]
    }

    # Store in recent history
    preview = text[:80] + ("…" if len(text) > 80 else "")
    RECENT_HISTORY.appendleft({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "preview": preview,
        "total_sentences": len(results),
        "counts": counts,
        "corrected_paragraph": corrected_paragraph
    })

    return jsonify(response_payload)

@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    if not question:
        return jsonify({"reply": "Ask a factual question or statement to verify."})

    # Grounded response: verify the question/statement directly
    evidence = retrieve_evidence(question)
    if evidence.get("evidence"):
        verif = verify_claim(question, evidence)
        src = evidence.get("source") or "Reference source"
        expl = verif.get("explanation") or ""
        # Avoid duplicating source citation if explanation already includes it
        if "Source:" in expl:
            reply = f"Claim status: {verif.get('status')} ({expl})"
        else:
            reply = f"Claim status: {verif.get('status')} ({expl}). Source: {src}."
        return jsonify({
            "status": verif.get("status"),
            "confidence": verif.get("confidence", 0.0),
            "explanation": expl,
            "source": src,
            "reply": reply
        })
    return jsonify({
        "status": "Unverified",
        "confidence": 0.0,
        "reply": "No definitive source evidence was found for that query. Please check spelling or query Wikipedia directly."
    })

@app.get("/api/history")
def history():
    return jsonify({"history": list(RECENT_HISTORY)})

@app.post("/api/history/clear")
@app.delete("/api/history")
def clear_history():
    RECENT_HISTORY.clear()
    return jsonify({"ok": True, "message": "History cleared successfully."})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
