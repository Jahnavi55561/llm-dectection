"""Evidence retrieval using a local knowledge base and Wikipedia's public API."""
import json
import os
import re
import time
import logging
import threading
import difflib
from typing import Dict, Any, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(ROOT, "knowledge_base.json")
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "LLMHallucinationFramework/1.0 (https://github.com/educational-prototype; contact: framework@local.net)"
}

STOP = set("""
a about above after again against all am an and any are aren't as at be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

PERSONAL_PATTERN = re.compile(
    r"\b(my\s+(neighbour|neighbor|friend|professor|college|class|house|teacher|family|boss|colleague|cousin|sister|brother|dad|mom|father|mother|dog|cat|car|laptop|phone|roommate|room)|"
    r"in\s+my\s+class|near\s+my\s+house|at\s+my\s+college|my\s+college\s+library|the\s+user\s+has|the\s+user\s+is|"
    r"a\s+student\s+in\s+my|a\s+person\s+named\s+[a-z]+)\b",
    re.IGNORECASE
)

VERB_SPLIT_PATTERN = re.compile(
    r"\b(is\s+commonly\s+used\s+to|is\s+generally|is\s+approximately|is\s+located\s+in|is\s+known\s+as|"
    r"is\s+the|is\s+an?|is|are|was\s+established\s+in|was\s+founded\s+in|was\s+discovered\s+in|"
    r"was|were|has|have|had|contains?|equals?|stands?\s+for|became|flows?\s+through|produces?|"
    r"breathes?|travels?\s+through|travels?|can\s+travel|freezes?\s+at|revolves?\s+around|orbits?|divides?|"
    r"normally\s+have|gained\s+independence\s+in|came\s+into\s+effect\s+on|lay|lays|laid)\b",
    re.IGNORECASE
)

# Bounded in-memory cache with thread lock
_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 3600
_MAX_CACHE_SIZE = 1000

# Cached knowledge base
_KB_CACHE = None
_KB_MTIME = 0

# Persistent HTTP Session with retries and connection pooling
_SESSION = requests.Session()
_retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
_SESSION.mount("https://", HTTPAdapter(max_retries=_retries, pool_connections=10, pool_maxsize=20))

def is_personal_claim(claim: str) -> bool:
    """Check if the claim refers to private, unindexed personal observations."""
    return bool(PERSONAL_PATTERN.search(claim))

def _tokens(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOP and len(w) > 1}

def _load_kb() -> List[Dict[str, Any]]:
    global _KB_CACHE, _KB_MTIME
    try:
        if os.path.exists(KB_PATH):
            mtime = os.path.getmtime(KB_PATH)
            if _KB_CACHE is not None and mtime == _KB_MTIME:
                return _KB_CACHE
            with open(KB_PATH, "r", encoding="utf-8") as f:
                _KB_CACHE = json.load(f)
                _KB_MTIME = mtime
                return _KB_CACHE
    except Exception as exc:
        logger.error("Failed to load knowledge base: %s", exc)
    return []

def _kb_lookup(claim: str) -> Optional[Dict[str, Any]]:
    claim_norm = re.sub(r"\W+", " ", claim.lower()).strip()
    if not claim_norm:
        return None

    best_match = None
    best_ratio = 0.0

    for item in _load_kb():
        if not isinstance(item, dict):
            continue
        cand = str(item.get("claim", ""))
        cand_norm = re.sub(r"\W+", " ", cand.lower()).strip()

        # Exact normalized match (word order preserved)
        if cand_norm and cand_norm == claim_norm:
            return {
                "evidence": str(item.get("evidence", "")),
                "source": str(item.get("source", "Local knowledge base")),
                "source_url": str(item.get("source_url", "")),
                "retrieval": "exact local knowledge-base match",
                "relevance_score": 1.0
            }

        # Sequence similarity to prevent matching inverted claims like "Sun revolves around Earth"
        ratio = difflib.SequenceMatcher(None, claim_norm, cand_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = item

    # High threshold requires word order and phrasing to closely match
    if best_match and best_ratio >= 0.88:
        return {
            "evidence": str(best_match.get("evidence", "")),
            "source": str(best_match.get("source", "Local knowledge base")),
            "source_url": str(best_match.get("source_url", "")),
            "retrieval": f"fuzzy local knowledge-base match ({round(best_ratio, 2)})",
            "relevance_score": round(best_ratio, 3)
        }

    return None

def _queries(claim: str) -> List[str]:
    q = claim.strip().rstrip(".!?")
    queries = []

    # 1. Extract subject before main verb/copula
    match = VERB_SPLIT_PATTERN.search(q)
    clean_subj = ""
    if match:
        subject = q[:match.start()].strip()
        clean_subj = re.sub(r"^(the|a|an|all|every|one)\s+", "", subject, flags=re.IGNORECASE).strip()
        if len(clean_subj) > 1:
            # Add singular form first for plural nouns (e.g. Bats -> Bat, Dolphins -> Dolphin)
            # Avoid stripping non-plural endings (e.g. Address, Glass, Cactus, Basis, Atlas)
            non_plural_endings = ("ss", "us", "is", "as")
            if (clean_subj.endswith("s") and len(clean_subj) >= 4
                    and not any(clean_subj.lower().endswith(end) for end in non_plural_endings)
                    and clean_subj.lower() not in {"css", "mars", "paris", "venus", "species", "air"}):
                singular = re.sub(r"ies$", "y", clean_subj)
                if singular == clean_subj:
                    singular = clean_subj[:-1]
                queries.insert(0, singular)

            queries.append(clean_subj)

            if "meter" in clean_subj.lower():
                queries.append(re.sub(r"meter", "metre", clean_subj, flags=re.IGNORECASE))
            elif "metre" in clean_subj.lower():
                queries.append(re.sub(r"metre", "meter", clean_subj, flags=re.IGNORECASE))

            if " of " in clean_subj.lower():
                part_after_of = clean_subj.split(" of ", 1)[-1].strip()
                if part_after_of:
                    queries.append(part_after_of)
            if " for " in clean_subj.lower():
                part_after_for = clean_subj.split(" for ", 1)[-1].strip()
                if part_after_for:
                    queries.append(part_after_for)

        # Composite queries for predicate nouns
        after_verb = q[match.end():].strip()
        after_words = [w for w in re.findall(r"[A-Za-z0-9'-]+", after_verb) if w.lower() not in STOP]
        if after_words:
            # Skip number words like three, four
            obj_words = [w for w in after_words if not re.match(r"^(one|two|three|four|five|six|seven|eight|nine|ten|\d+)$", w, re.I)]
            if obj_words:
                main_obj = obj_words[0]
                if clean_subj:
                    queries.append(f"{clean_subj} {main_obj}")
                queries.append(main_obj)
                if (main_obj.endswith("s") and len(main_obj) > 3
                        and not any(main_obj.lower().endswith(end) for end in ("ss", "us", "is", "as"))):
                    queries.append(main_obj[:-1])

    # 2. Number entity pattern (e.g. "number 21" -> "21 (number)")
    num_match = re.search(r"\bnumber\s+(\d+)\b", q, flags=re.IGNORECASE)
    if num_match:
        queries.append(f"{num_match.group(1)} (number)")

    # 3. Chemical element symbols
    chem_match = re.search(r"chemical symbol for\s+([A-Za-z]+)", q, flags=re.IGNORECASE)
    if chem_match:
        queries.append(chem_match.group(1).capitalize())

    # 4. Anatomical organs and body features
    organ_match = re.search(r"\b(?:have|has|having|contain|contains|with)\b.*?\b(lungs?|bones?|hearts?|chambers?|bloods?)\b", q, re.I)
    if organ_match:
        organ = organ_match.group(1).lower()
        if organ.startswith("lung"):
            queries.insert(0, "Lung")
            queries.insert(1, "Human lung")
        elif organ.startswith("bone"):
            queries.insert(0, "Human skeleton")
            queries.insert(1, "Bone")
        elif organ.startswith("chamber") or organ.startswith("heart"):
            queries.insert(0, "Heart")
        elif organ.startswith("blood"):
            queries.insert(0, "Blood")
            queries.insert(1, "Human blood")

    # 5. Computing acronyms and disambiguations
    q_lower = q.lower()
    if "ipv4" in q_lower:
        queries.insert(0, "IPv4")
    elif "ipv6" in q_lower:
        queries.insert(0, "IPv6")
    elif re.search(r"\bram\b", q_lower) and "memory" in q_lower:
        queries.insert(0, "Random-access memory")

    # 6. Capitalized Named Entities (proper nouns)
    entities = re.findall(r"\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b", q)
    for ent in entities:
        ent = ent.strip()
        if ent.lower() not in {"the", "a", "an", "all", "every", "one", "zero", "first", "scientists", "researchers", "nasa"}:
            if ent not in queries:
                queries.append(ent)

    # 7. Compact keyword query
    words = [w for w in re.findall(r"[A-Za-z0-9'-]+", q) if w.lower() not in STOP]
    if len(words) >= 2:
        queries.append(" ".join(words[:6]))

    # 8. Full query
    queries.append(q)

    # Deduplicate while preserving order
    seen, out = set(), []
    for item in queries:
        clean = item.strip()
        if clean and clean.lower() not in seen and len(clean) > 1:
            out.append(clean)
            seen.add(clean.lower())
    return out


def _wiki_search(query: str) -> List[Dict[str, Any]]:
    for attempt in range(2):
        try:
            response = _SESSION.get(
                API,
                params={
                    "action": "query", "list": "search", "srsearch": query,
                    "srlimit": 4, "format": "json", "utf8": 1
                },
                headers=HEADERS,
                timeout=8
            )
            response.raise_for_status()
            return response.json().get("query", {}).get("search", [])
        except Exception as exc:
            if attempt == 0:
                time.sleep(1.0)
                continue
            logger.warning("Wikipedia search failed for query '%s': %s", query, exc)
            return []
    return []

def _wiki_extracts_batch(pageids: List[int]) -> Dict[int, Dict[str, str]]:
    if not pageids:
        return {}
    pageids_str = "|".join(str(pid) for pid in pageids[:10])
    for attempt in range(2):
        try:
            response = _SESSION.get(
                API,
                params={
                    "action": "query", "pageids": pageids_str, "prop": "extracts|info",
                    "explaintext": 1, "exintro": 1, "inprop": "url",
                    "format": "json", "redirects": 1
                },
                headers=HEADERS,
                timeout=10
            )
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", {})
            results = {}
            for pid_str, page_data in pages.items():
                try:
                    pid = int(pid_str)
                    extract = page_data.get("extract", "")
                    url = page_data.get("fullurl", "")
                    results[pid] = {"extract": extract, "url": url}
                except (ValueError, TypeError):
                    continue
            return results
        except Exception as exc:
            if attempt == 0:
                time.sleep(1.0)
                continue
            logger.warning("Wikipedia batch extract failed: %s", exc)
            return {}
    return {}


def retrieve_evidence(claim: str) -> Dict[str, Any]:
    key = claim.strip().lower()
    now = time.time()

    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and (now - cached[0] < _CACHE_TTL):
            return dict(cached[1])

    # 1. Personal claim guard: private personal assertions cannot be verified by public encyclopedias
    if is_personal_claim(claim):
        res = {
            "evidence": "",
            "source": "",
            "source_url": "",
            "retrieval": "personal_claim",
            "relevance_score": 0.0
        }
        with _CACHE_LOCK:
            _CACHE[key] = (now, res)
        return res

    # 2. Local Knowledge Base lookup
    local = _kb_lookup(claim)
    if local and local.get("evidence"):
        with _CACHE_LOCK:
            _CACHE[key] = (now, local)
        return dict(local)

    # 3. Gather candidate Wikipedia pages from generated queries
    query_results = {}
    queries = _queries(claim)
    primary_subject = queries[0].lower() if queries else ""
    claim_lower = claim.lower()

    for query in queries[:5]:
        for result in _wiki_search(query):
            pageid = result.get("pageid")
            if pageid and pageid not in query_results:
                query_results[pageid] = result

    # 4. Batch retrieve extracts
    all_pageids = list(query_results.keys())[:10]
    page_extracts = _wiki_extracts_batch(all_pageids)

    claim_tokens = _tokens(claim)
    candidates = []

    for pageid, result in query_results.items():
        extract_info = page_extracts.get(pageid)
        if not extract_info:
            continue
        extract = extract_info.get("extract", "").strip()
        url = extract_info.get("url", "")
        if not extract:
            continue

        title = result.get("title", "")
        title_lower = title.lower()
        if "(disambiguation)" in title_lower:
            continue

        ev_tokens = _tokens(title + " " + extract[:3500])
        title_tokens = _tokens(title)

        if not claim_tokens:
            continue

        overlap = len(claim_tokens & ev_tokens) / len(claim_tokens)
        title_boost = 0.20 if (claim_tokens & title_tokens) else 0.0

        # Exact title match to subject query gets top priority
        is_exact = any(title_lower == q.lower() for q in queries[:3])
        subject_boost = 0.55 if is_exact else (0.25 if primary_subject and primary_subject in title_lower else 0.0)

        # Penalize unrelated sports teams, films, and novels when claim does not ask for them
        penalized_terms = [
            "cleveland", "pittsburgh", "miami", "nfl", "nba", "championship",
            "fc", "club", "team", "murderer", "film", "album", "novel", "band",
            "football", "baseball", "basketball", "sports", "franchise", "rugby", "league", "nrl"
        ]
        has_penalty = any(term in title_lower for term in penalized_terms) and not any(term in claim_lower for term in penalized_terms)
        sports_penalty = -0.45 if has_penalty else 0.0
        list_penalty = -0.25 if title_lower.startswith("list of") else 0.0

        # Penalize uppercase acronym title (e.g. BATS) if query was a lowercase common word
        is_acronym_mismatch = (title.isupper() and len(title) >= 3 and not any(w.isupper() and len(w) >= 3 for w in claim.split()))
        acronym_penalty = -0.40 if is_acronym_mismatch else 0.0

        score = overlap + title_boost + subject_boost + sports_penalty + acronym_penalty + list_penalty
        candidates.append((score, title, extract, url))

    candidates.sort(key=lambda x: x[0], reverse=True)

    # Minimum relevance threshold
    if candidates and candidates[0][0] >= 0.18:
        score, title, extract, url = candidates[0]
        # If extract is very short (e.g. number pages like 21), fetch full body text
        if len(extract) < 300:
            try:
                single_resp = _SESSION.get(
                    API,
                    params={
                        "action": "query", "titles": title, "prop": "extracts",
                        "explaintext": 1, "format": "json", "redirects": 1
                    },
                    headers=HEADERS,
                    timeout=8
                )
                if single_resp.ok:
                    pages = single_resp.json().get("query", {}).get("pages", {})
                    for p in pages.values():
                        if p.get("extract"):
                            extract = p.get("extract")
                            break
            except Exception:
                pass

        result = {
            "evidence": extract[:4500],
            "source": f"Wikipedia — {title}",
            "source_url": url,
            "retrieval": "Wikipedia API",
            "relevance_score": round(score, 3)
        }
    else:
        result = {
            "evidence": "",
            "source": "",
            "source_url": "",
            "retrieval": "no relevant evidence found",
            "relevance_score": 0
        }


    with _CACHE_LOCK:
        if len(_CACHE) >= _MAX_CACHE_SIZE:
            old_keys = sorted(_CACHE.keys(), key=lambda k: _CACHE[k][0])[: len(_CACHE) // 4]
            for k in old_keys:
                _CACHE.pop(k, None)
        _CACHE[key] = (now, result)

    return dict(result)

def find_evidence(claim: str) -> Dict[str, Any]:
    return retrieve_evidence(claim)



