# pipeline.py (drop-in replacement)
from typing import Dict, Any, List
from pathlib import Path
import json, re, math

from embed_index import search_topk
from ranker import score_candidates

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
CLARIFIER_PATH = ARTIFACTS_DIR / "clarifier.json"
DOCS_PATH = ARTIFACTS_DIR / "docs.json"

# thresholds (you can tweak)
TOP1_DIRECT_ANSWER = 0.75  # >= -> answer directly
TOP1_MIN_IN_DOMAIN = 0.60  # <  -> likely out-of-scope
MIN_TOKENS = 2             # too-short queries are usually small talk

def _load_margin_t() -> float:
    try:
        obj = json.loads(CLARIFIER_PATH.read_text(encoding="utf-8"))
        return float(obj.get("margin_t", 0.1))
    except Exception:
        return 0.1

def _load_docs() -> List[Dict[str,Any]]:
    try:
        return json.loads(DOCS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def _greeting_or_smalltalk(q: str) -> bool:
    return bool(re.match(r"^\s*(hi|hello|hey|yo|sup|good\s*(morning|afternoon|evening))\b", q.strip(), re.I))

def _any_trigger_overlap(q: str, doc: Dict[str,Any]) -> bool:
    trig = doc.get("triggers","")
    if not trig:
        return True
    qtok = set(_tokenize(q))
    for t in re.split(r"[,\|/;:\n]+", trig):
        t = t.strip().lower()
        if not t:
            continue
        if set(_tokenize(t)) & qtok:
            return True
    return False

def _softmax(xs, tau=0.1):
    m = max(xs)
    exps = [math.exp((x - m)/tau) for x in xs]
    s = sum(exps) or 1.0
    return [e/s for e in exps]

def _entropy(ps):
    return -sum(p * math.log(p + 1e-12) for p in ps)

def _oos_message() -> Dict[str,Any]:
    msg = (
        "I’m for Bishop’s University IT support (Wi-Fi, Moodle, email, MyBU, accounts, printing). "
        "Please ask an IT-related question or call 819-822-9600 ext. 2273."
    )
    return {"type":"answer","intent_id":None,"intent_name":"Out-of-scope","steps":msg,
            "primary_link":"","secondary_link":"","rank_score":0.0,"alternatives":[]}

def _refuse_message() -> Dict[str,Any]:
    msg = (
        "I can’t help with hacking, credential theft, or bypassing security. "
        "If you need legitimate access help (e.g., password reset, account recovery), I can guide you."
    )
    return {"type":"answer","intent_id":None,"intent_name":"Safety","steps":msg,
            "primary_link":"","secondary_link":"","rank_score":0.0,"alternatives":[]}

def answer(query: str) -> Dict[str, Any]:
    q = (query or "").strip()
    docs = _load_docs()
    if not q:
        return _oos_message()

    # 0) safety & small talk
    if any(w in q.lower() for w in ["hack", "bypass", "steal", "phish", "exploit"]):
        return _refuse_message()
    if _greeting_or_smalltalk(q) or len(_tokenize(q)) < MIN_TOKENS:
        return _oos_message()

    # 1) retrieve
    cands = search_topk(q, k=5)
    if not cands:
        return _oos_message()

    # 2) simple in-domain gate by top1 + trigger overlap
    top1 = cands[0]["score"]
    has_overlap = any(_any_trigger_overlap(q, c) for c in cands[:3])
    if (top1 < TOP1_MIN_IN_DOMAIN) and (not has_overlap):
        return _oos_message()

    # 3) clarification: only when margin small AND confidence mid-range AND distribution uncertain
    margin_t = _load_margin_t()
    margin = cands[0]["score"] - (cands[1]["score"] if len(cands) > 1 else 0.0)
    sims = [c["score"] for c in cands[:3]]
    ps = _softmax(sims, tau=0.08)
    H = _entropy(ps)
    need_clarify = (TOP1_MIN_IN_DOMAIN <= top1 < TOP1_DIRECT_ANSWER) and (margin <= margin_t) and (H > 0.8)

    if need_clarify:
        options = [{"intent_id": c["intent_id"], "intent_name": c["intent_name"]} for c in cands[:3]]
        return {"type":"clarify",
                "message":"Your question may refer to multiple intents. Reply with a number to confirm:",
                "options": options}

    # 4) direct answer (rank candidates for quality)
    ranked = score_candidates(q, cands)
    top = ranked[0]
    return {
        "type": "answer",
        "intent_id": top["intent_id"],
        "intent_name": top["intent_name"],
        "steps": top["steps"],
        "primary_link": top.get("primary_link",""),
        "secondary_link": top.get("secondary_link",""),
        "rank_score": top["rank_score"],
        "alternatives": [{"intent_name": r["intent_name"], "rank_score": r["rank_score"]} for r in ranked[1:3]]
    }
