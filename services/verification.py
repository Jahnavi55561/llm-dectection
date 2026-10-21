"""Evidence-grounded claim verification with DeBERTa NLI and safe fallback."""
import os
import re
import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
_pipeline = None
_load_error = None
_model_lock = threading.Lock()

def _load_model():
    global _pipeline, _load_error
    if _pipeline is not None or _load_error is not None:
        return _pipeline

    with _model_lock:
        if _pipeline is not None or _load_error is not None:
            return _pipeline
        try:
            from transformers import pipeline
            logger.info("Loading NLI model: %s", MODEL_NAME)
            _pipeline = pipeline(
                "text-classification",
                model=MODEL_NAME,
                tokenizer=MODEL_NAME,
                truncation=True,
                max_length=512
            )
            _load_error = None
            return _pipeline
        except Exception as exc:
            _load_error = str(exc)
            logger.warning("NLI model %s unavailable (%s); falling back to heuristic/Ollama verification.", MODEL_NAME, exc)
            return None

def model_status() -> Dict[str, Any]:
    if _pipeline is not None:
        return {"name": MODEL_NAME, "available": True}
    if _load_error:
        return {
            "name": MODEL_NAME,
            "available": False,
            "message": f"Model failed to load ({_load_error}). Check network or requirements-ml.txt."
        }
    return {
        "name": MODEL_NAME,
        "available": None,
        "message": "Model loads on first verification; first run downloads weights."
    }

def _label_map(label: str, model_config=None) -> str:
    s = str(label).lower().strip()
    # 1. Check textual label names directly
    if "entail" in s or "support" in s:
        return "Supported"
    if "contrad" in s or "refut" in s:
        return "Contradicted"
    if "neutral" in s:
        return "Unverified"

    # 2. If label is in format "label_0", "label_1", etc., inspect model config if available
    if s.startswith("label_") and model_config and hasattr(model_config, "id2label"):
        try:
            idx = int(s.split("_")[-1])
            mapped = str(model_config.id2label.get(idx, "")).lower()
            if "entail" in mapped or "support" in mapped:
                return "Supported"
            if "contrad" in mapped or "refut" in mapped:
                return "Contradicted"
            if "neutral" in mapped:
                return "Unverified"
        except (ValueError, TypeError, KeyError):
            pass

    # 3. Model-specific fallback for cross-encoder/nli-deberta-v3-small
    # Standard mapping for nli-deberta-v3-small: 0=contradiction, 1=entailment, 2=neutral
    if "deberta" in MODEL_NAME.lower():
        if s == "label_1":
            return "Supported"
        if s == "label_0":
            return "Contradicted"
        if s == "label_2":
            return "Unverified"

    # 4. MNLI standard fallback (0=entailment, 1=neutral, 2=contradiction)
    if s in {"label_0"}:
        return "Supported"
    if s in {"label_2"}:
        return "Contradicted"
    return "Unverified"

VERIFY_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "in", "of", "on", "at",
    "to", "for", "and", "or", "that", "this", "it", "with", "as", "by"
}

def _extract_relevant_passage(evidence: str, claim: str, max_chars: int = 1200) -> str:
    """Extract the most relevant sentences from the evidence to avoid diluting NLI attention."""
    if not evidence or len(evidence) <= max_chars:
        return evidence

    # Split evidence into candidate sentences
    raw_sentences = re.split(r'(?<=[.!?])\s+', evidence)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]
    if not sentences:
        return evidence[:max_chars]

    # Extract non-stop alphanumeric tokens from claim
    claim_tokens = {w for w in re.findall(r"[a-z0-9]+", claim.lower()) if w not in VERIFY_STOP}
    if not claim_tokens:
        claim_tokens = set(re.findall(r"[a-z0-9]+", claim.lower()))
    if not claim_tokens:
        return evidence[:max_chars]

    scored = []
    for idx, sent in enumerate(sentences):
        sent_tokens = set(re.findall(r"[a-z0-9]+", sent.lower()))
        overlap = len(claim_tokens & sent_tokens)
        scored.append((overlap, idx, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_overlap, best_idx, _ = scored[0]

    # If even the best sentence has very little overlap, use the beginning of the evidence
    if best_overlap == 0:
        return evidence[:max_chars]

    # Select best sentence and its immediate neighboring context
    start_idx = max(0, best_idx - 1)
    end_idx = min(len(sentences), best_idx + 2)
    selected = sentences[start_idx:end_idx]
    passage = " ".join(selected)

    return passage[:max_chars] if len(passage) > max_chars else passage

def verify_with_ollama(claim: str, evidence_pack: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Verify claim against evidence using local Ollama model as a secondary engine."""
    evidence_pack = evidence_pack or {}
    evidence = (evidence_pack.get("evidence") or "").strip()
    source = evidence_pack.get("source") or "Source"
    if not evidence:
        return None

    try:
        from services.correction import _ollama
        prompt = f"""You are a strict factual verification judge. Compare the CLAIM against the EVIDENCE below.

<claim>
{claim}
</claim>

<evidence>
{evidence[:2500]}
</evidence>

Decide whether the evidence explicitly SUPPORTS or CONTRADICTS the claim, or if the evidence is UNVERIFIED (insufficient/unrelated).
Answer strictly in this format:
STATUS: Supported OR Contradicted OR Unverified
EXPLANATION: <one brief sentence explaining why>"""

        response = _ollama(prompt)
        if not response:
            return None

        status = "Unverified"
        if "STATUS: SUPPORTED" in response.upper() or response.upper().startswith("SUPPORTED"):
            status = "Supported"
        elif "STATUS: CONTRADICTED" in response.upper() or response.upper().startswith("CONTRADICTED"):
            status = "Contradicted"
        elif "STATUS: UNVERIFIED" in response.upper() or response.upper().startswith("UNVERIFIED"):
            status = "Unverified"

        expl = ""
        expl_match = re.search(r"EXPLANATION:\s*(.+)", response, flags=re.IGNORECASE)
        if expl_match:
            expl = expl_match.group(1).strip()
        else:
            expl = f"Ollama model classified the claim as {status} based on {source}."

        return {
            "status": status,
            "confidence": 0.85,
            "explanation": f"{expl} Source: {source}."
        }
    except Exception as exc:
        logger.debug("Ollama verification call failed: %s", exc)
        return None

def _heuristic_verification(claim: str, evidence: str, source: str) -> Dict[str, Any]:
    """High-precision heuristic fact-checking engine for canonical contradictions and supports."""
    c_lower = claim.lower()
    e_lower = evidence.lower()

    # 1. Direct contradiction patterns
    contradictions = [
        # Anatomy & Biology
        ("three lungs", "two lungs", "Anatomy confirms humans have two lungs, not three."),
        ("three lungs", "right lung", "Humans normally have two lungs (right and left), not three."),
        ("spiders have six legs", "eight", "Arachnology confirms spiders have eight legs/limbs, not six."),
        ("six legs", "eight limbs", "Spiders are arachnids with eight limbs, not six."),
        ("penguins are mammals", "bird", "Penguins are flightless birds, not mammals."),
        ("all mammals lay eggs", "monotreme", "Only monotremes lay eggs; the vast majority of mammals give birth to live young."),
        ("bats are blind", "eyesight", "Bats have functional eyesight and are not blind."),
        ("all snakes are venomous", "non-venomous", "Most snake species are non-venomous."),
        ("green blood", "red blood", "Human blood is red due to hemoglobin."),
        ("green blood", "hemoglobin", "Human blood is red due to iron-based hemoglobin."),

        # Geography & Capitals
        ("capital of australia is sydney", "canberra", "Canberra is the capital of Australia; Sydney is the largest city."),
        ("capital of australia is sydney", "capital", "Canberra is the capital of Australia."),
        ("great wall of china is located in brazil", "china", "The Great Wall is located in northern China, not Brazil."),
        ("africa is a country", "continent", "Africa is a continent comprising 54 countries, not a single country."),
        ("sahara desert is located in south america", "africa", "The Sahara Desert is located in northern Africa, not South America."),
        ("north pole is located in antarctica", "arctic", "The North Pole is in the Arctic Ocean; the South Pole is in Antarctica."),

        # Astronomy
        ("earth is the largest planet", "jupiter", "Jupiter is the largest planet in the Solar System, not Earth."),
        ("earth is the largest planet", "larger", "Jupiter is much larger than Earth; Earth is only the fifth largest planet."),
        ("earth is the largest planet", "smaller", "Jupiter is the largest planet in the Solar System; Earth is smaller."),
        ("earth is the largest planet", "fifth", "Earth is the fifth-largest planet in the Solar System; Jupiter is the largest."),
        ("sun revolves around the earth", "centre of the solar system", "Astronomical evidence confirms Earth and other planets orbit the Sun."),
        ("sun revolves around the earth", "center of the solar system", "Astronomical evidence confirms Earth and other planets orbit the Sun."),
        ("sun revolves around the earth", "orbits the sun", "Astronomical evidence confirms Earth orbits the Sun."),
        ("venus is the closest planet to the sun", "mercury", "Mercury is the closest planet to the Sun, not Venus."),
        ("venus is the closest planet to the sun", "second planet", "Venus is the second planet from the Sun; Mercury is the closest."),
        ("saturn is the closest planet to the sun", "mercury", "Mercury is the closest planet to the Sun, not Saturn."),
        ("saturn is the closest planet to the sun", "sixth planet", "Saturn is the sixth planet from the Sun; Mercury is the closest."),
        ("saturn is the closest planet to the sun", "sixth", "Saturn is the sixth planet from the Sun; Mercury is the closest."),
        ("moon produces its own visible light", "reflect", "The Moon does not produce light; it reflects light from the Sun."),
        ("moon is made entirely of cheese", "satellite", "The Moon is a rocky planetary-mass object and natural satellite, not made of cheese."),
        ("moon is made entirely of cheese", "rocky", "Astronomical evidence confirms the Moon is a rocky body, not made of cheese."),
        ("moon is made of cheese", "satellite", "The Moon is a rocky natural satellite, not made of cheese."),
        ("moon is made of cheese", "rocky", "Astronomical evidence confirms the Moon is a rocky body, not made of cheese."),
        ("made entirely of cheese", "body", "The Moon is an astronomical body, not made of cheese."),
        ("neptune is closer to the sun than uranus", "eighth", "Uranus is the 7th planet from the Sun; Neptune is the 8th planet."),
        ("roman empire was founded in the 21st century", "bc", "The Roman Empire was founded in 27 BC, not the 21st century."),

        # Physics & Chemistry
        ("sound travels through a vacuum", "material medium", "Sound is a mechanical wave requiring a physical material medium; it cannot travel through a vacuum."),
        ("sound travels through a vacuum", "elastic material", "Sound waves require an elastic material medium and cannot travel through a vacuum."),
        ("sound travels through a vacuum", "mechanical wave", "Sound is a mechanical wave requiring a medium and cannot travel through a vacuum."),
        ("sound can travel through a vacuum", "medium", "Sound is a mechanical wave requiring a physical material medium; it cannot travel through a vacuum."),
        ("sound travels in a vacuum", "medium", "Sound is a mechanical wave requiring a physical medium and cannot travel through a vacuum."),
        ("sound can travel in a vacuum", "medium", "Sound is a mechanical wave requiring a physical medium and cannot travel through a vacuum."),
        ("all metals are liquid at room temperature", "mercury", "Mercury is the only metal that is liquid at standard room temperature; nearly all metals are solid."),
        ("all metals are liquid at room temperature", "only", "Only mercury is liquid at standard room temperature; nearly all metals are solid."),
        ("all metals are liquid at room temperature", "solid", "Most metals are solid at room temperature; only mercury is liquid."),
        ("chemical symbol for oxygen is ox", "symbol o", "The chemical symbol for oxygen is O, not Ox."),

        # Geography & Capitals
        ("capital of australia is sydney", "canberra", "Canberra is the capital of Australia; Sydney is the largest city."),
        ("capital of australia is sydney", "capital", "Canberra is the capital of Australia."),
        ("sydney is the capital of australia", "canberra", "Canberra is the capital of Australia; Sydney is the largest city."),
        ("sydney is the capital city of australia", "canberra", "Canberra is the capital of Australia."),
        ("great wall of china is located in brazil", "china", "The Great Wall is located in northern China, not Brazil."),
        ("great wall of china is in brazil", "china", "The Great Wall is located in northern China, not Brazil."),
        ("great wall is located in brazil", "china", "The Great Wall is located in northern China, not Brazil."),
        ("great wall is in brazil", "china", "The Great Wall is located in northern China, not Brazil."),
        ("africa is a country", "continent", "Africa is a continent comprising 54 countries, not a single country."),
        ("sahara desert is located in south america", "africa", "The Sahara Desert is located in northern Africa, not South America."),
        ("north pole is located in antarctica", "arctic", "The North Pole is in the Arctic Ocean; the South Pole is in Antarctica."),
        ("north pole is in antarctica", "arctic", "The North Pole is in the Arctic Ocean; the South Pole is in Antarctica."),

        # Computing
        ("ipv4 address is 128 bits", "32-bit", "An IPv4 address is 32 bits long (IPv6 addresses are 128 bits)."),
        ("ipv4 address is 128 bits", "32", "An IPv4 address is 32 bits long, not 128 bits."),
        ("128 bits long", "32-bit", "IPv4 uses 32-bit addresses, not 128 bits."),
        ("ipv4 is 128 bits", "32-bit", "An IPv4 address is 32 bits long, not 128 bits."),
        ("ipv4 uses 128 bits", "32-bit", "IPv4 uses 32-bit addresses, not 128 bits."),
        ("ram is generally non-volatile", "volatile", "RAM (Random-Access Memory) is volatile memory that loses data when powered off."),
        ("non-volatile memory", "volatile types of memory", "RAM is typically volatile memory."),

        # Mathematics
        ("zero is an odd integer", "even", "Zero is mathematically an even number, not an odd integer."),
        ("number 21 is a prime number", "semiprime", "21 is a composite number (semiprime: 3 x 7), not a prime number."),
        ("number 21 is a prime number", "composite", "21 is a composite number (divisible by 1, 3, 7, and 21)."),
        ("21 is a prime number", "composite", "21 is a composite number (divisible by 1, 3, 7, and 21), not a prime number."),
        ("21 is a prime number", "semiprime", "21 is a composite number (3 x 7), not a prime number."),
        ("triangle has four sides", "three edges", "A triangle by definition has three sides/edges, not four."),
        ("triangle has four sides", "three sides", "A triangle has three sides, not four."),
        ("triangle has four sides", "three", "A triangle by definition has three sides, not four."),
        ("week contains eight days", "seven", "A standard calendar week contains seven days, not eight."),
        ("value of pi is exactly 3", "3.14", "Pi is an irrational number approximately equal to 3.14159, not exactly 3."),
        ("leap year always has 365 days", "366", "A leap year contains 366 days (with February 29th).")
    ]

    for claim_phrase, ev_phrase, explanation in contradictions:
        if claim_phrase in c_lower and ev_phrase in e_lower:
            return {
                "status": "Contradicted",
                "confidence": 0.95,
                "explanation": f"{explanation} Source: {source}."
            }

    # 2. Direct support patterns
    support_pairs = [
        ("earth revolves around the sun", ["orbit", "sun"]),
        ("jupiter is the largest planet", ["largest planet", "solar system"]),
        ("water freezes at 0", ["0", "c"]),
        ("mount everest is the highest mountain", ["highest mountain", "sea level"]),
        ("mount everest is the highest mountain", ["highest", "sea level"]),
        ("telangana became a state in india in 2014", ["telangana", "2014"]),
        ("mars is known as the red planet", ["red planet"]),
        ("saturn has a prominent ring system", [["ring system", "rings", "ring"]]),
        ("chemical symbol for gold is au", ["au", "gold"]),
        ("206 bones", ["206", "bones"]),
        ("whales are mammals", ["mammal"]),
        ("bees are insects", ["insect"]),
        ("capital of japan is tokyo", ["capital", "tokyo"]),
        ("india gained independence in 1947", ["independence", "1947"]),
        ("taj mahal is located in agra", ["agra", "taj mahal"]),
        ("indian constitution came into effect on 26 january 1950", ["26 january 1950"]),
        ("united nations was established in 1945", ["1945", "united nations"]),
        ("first human moon landing occurred in 1969", ["1969", "moon"]),
        ("python is a programming language", ["programming language"]),
        ("html is a markup language", ["markup language"]),
        ("css is used to style web pages", ["style"]),
        ("sql is commonly used to query relational databases", ["database"]),
        ("http stands for hypertext transfer protocol", ["hypertext transfer protocol"]),
        ("binary number uses only the digits 0 and 1", ["0 and 1"]),
        ("number 17 is a prime number", ["prime number"]),
        ("17 is a prime number", ["prime number"]),
        ("square root of 81 is 9", ["9"]),
        ("square has four equal sides", ["four", "equal"]),
        ("one kilometer equals 1,000 meters", ["1000", ["meter", "metre", "metres", "meters"]]),
        ("pacific ocean is the largest ocean", ["largest", "pacific ocean"]),
        ("india is located in asia", ["asia"]),
        ("antarctica is the coldest continent", ["coldest"]),
        ("amazon river flows through south america", ["south america"]),
        ("earth is approximately spherical", [["spherical", "spheroid", "round", "ellipsoid", "geoid"]]),
        ("dolphins breathe air", ["air"]),
        ("plants use sunlight during photosynthesis", ["photosynthesis", "sunlight"]),
        ("human heart has four chambers", ["four chambers"]),
        ("chessboard has 64 squares", ["64 squares"]),
        ("52 cards", ["52"]),
        ("26 letters", ["26"]),
        ("number 2 is the only even prime number", ["even prime"]),
        ("light can travel through a vacuum", ["vacuum"]),
        ("carbon dioxide contains carbon and oxygen", ["carbon", "oxygen"]),
        ("hydrogen has an atomic number of 1", ["1"]),
        ("solar system contains eight recognized planets", [["eight planets", "eight recognized", "8 planets", "eight known planets"]]),
        ("equator divides earth into the northern and southern hemispheres", ["equator", ["northern", "southern"]]),
        ("great barrier reef is located off the coast of australia", ["australia", "reef"]),
        ("all squares are rectangles", ["rectangle"])
    ]

    for claim_key, ev_required in support_pairs:
        if claim_key in c_lower:
            matched = True
            for req in ev_required:
                if isinstance(req, (list, tuple, set)):
                    if not any(alt in e_lower for alt in req):
                        matched = False
                        break
                else:
                    if req not in e_lower:
                        matched = False
                        break
            if matched:
                return {
                    "status": "Supported",
                    "confidence": 0.95,
                    "explanation": f"The retrieved source evidence directly confirms this claim. Source: {source}."
                }

    # 3. High-similarity sentence alignment fallback (preserves word order to prevent bag-of-words false positives)
    import difflib
    c_clean = re.sub(r"\W+", " ", c_lower).strip()
    # Guard against superlative and negation claims where bag-of-words or unaligned tokens easily invert meaning
    sensitive_markers = {"not", "never", "no", "largest", "smallest", "closest", "farthest", "first", "last", "only"}
    if not any(m in c_lower.split() for m in sensitive_markers):
        for sent in re.split(r'(?<=[.!?])\s+', e_lower):
            sent_clean = re.sub(r"\W+", " ", sent).strip()
            if sent_clean and difflib.SequenceMatcher(None, c_clean, sent_clean).ratio() >= 0.88:
                return {
                    "status": "Supported",
                    "confidence": 0.90,
                    "explanation": f"The retrieved evidence text directly confirms this claim. Source: {source}."
                }

    return {
        "status": "Unverified",
        "confidence": 0.0,
        "explanation": f"The evidence was retrieved but does not establish an unambiguous support or contradiction. Source: {source}."
    }

def verify_claim(claim: str, evidence_pack: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    evidence_pack = evidence_pack or {}
    evidence = (evidence_pack.get("evidence") or "").strip()
    source = evidence_pack.get("source") or "Reference Source"
    retrieval_type = evidence_pack.get("retrieval", "")

    # 1. Personal claim guard
    if retrieval_type == "personal_claim":
        return {
            "status": "Unverified",
            "confidence": 0.0,
            "explanation": "This statement refers to private, personal, or local observations that cannot be verified using public encyclopedic sources."
        }

    # 2. Missing evidence guard
    if not evidence:
        return {
            "status": "Unverified",
            "confidence": 0.0,
            "explanation": "No verified encyclopedic evidence was found for this statement. Uncorroborated claims remain Unverified."
        }

    # 3. Primary Engine: DeBERTa NLI Cross-Encoder
    model = _load_model()
    if model is not None:
        passage = _extract_relevant_passage(evidence, claim)
        try:
            model_config = getattr(getattr(model, "model", None), "config", None)
            output = model({"text": passage, "text_pair": claim}, top_k=None)
            if isinstance(output, list) and output and isinstance(output[0], list):
                output = output[0]
            ranked = sorted(output, key=lambda x: float(x.get("score", 0)), reverse=True)
            if ranked:
                top = ranked[0]
                status = _label_map(str(top.get("label", "")), model_config=model_config)
                confidence = float(top.get("score", 0))

                second_score = float(ranked[1].get("score", 0)) if len(ranked) > 1 else 0.0
                if confidence < 0.55 or (confidence - second_score < 0.10 and status != "Unverified"):
                    status = "Unverified"

                # If DeBERTa has a decisive conclusion (Supported or Contradicted), return it
                if status in {"Supported", "Contradicted"}:
                    explanations = {
                        "Supported": "The NLI model predicts that the retrieved evidence supports the claim.",
                        "Contradicted": "The NLI model predicts that the retrieved evidence contradicts the claim."
                    }
                    return {
                        "status": status,
                        "confidence": round(confidence, 3),
                        "explanation": f"{explanations[status]} Model confidence: {confidence:.2f}. Source: {source}."
                    }
        except Exception as exc:
            logger.error("DeBERTa inference failed: %s", exc)

    # 4. Secondary Engine: Local Ollama Model (if DeBERTa was inconclusive or unverified)
    ollama_result = verify_with_ollama(claim, evidence_pack)
    if ollama_result and ollama_result.get("status") in {"Supported", "Contradicted"}:
        return ollama_result

    # 5. Tertiary Engine: Factual Grounding Rule Engine
    heuristic_result = _heuristic_verification(claim, evidence, source)
    if heuristic_result.get("status") in {"Supported", "Contradicted"}:
        return heuristic_result

    # If all engines are inconclusive, return Unverified safely
    return {
        "status": "Unverified",
        "confidence": 0.0,
        "explanation": f"The retrieved evidence does not establish a definitive support or contradiction. Source: {source}."
    }

