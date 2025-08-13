import os
import time
import re
import json
import pandas as pd
import streamlit as st
from groq import Groq

# pipeline
from pipeline import answer as pipeline_answer

# ---------------- Page ----------------
st.set_page_config(page_title="Bishop's IT Support Chatbot", page_icon="💻", layout="wide")
st.title("Bishop's University IT Support Chatbot")
st.markdown("We're here to help with Wi-Fi, Moodle, email, and more!")

# ---------------- API key ----------------
GROQ_API_KEY = (
    os.getenv("GROQ_API_KEY")
    or (st.secrets["GROQ_API_KEY"] if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets else None)
    or "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"  # replace via secrets in prod
)

def get_client() -> Groq:
    if "groq_client" not in st.session_state:
        st.session_state["groq_client"] = Groq(api_key=GROQ_API_KEY)
    return st.session_state["groq_client"]

# ---------------- Ground truth -> system prompt ----------------
SYSTEM_PROMPT_FALLBACK = (
    "You are Bishop’s University Level-0 IT Support Assistant. "
    "Answer only for Bishop’s students/staff. Be concise. "
    "Use 3–6 numbered steps. If Level-0 steps fail, escalate to the Helpdesk at 819-822-9600 ext. 2273."
)

def load_gt(path: str = "enhanced_ground_truth.csv") -> pd.DataFrame | None:
    try:
        return pd.read_csv(path) if os.path.exists(path) else None
    except Exception:
        return None

def build_system_prompt_from_gt(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return SYSTEM_PROMPT_FALLBACK
    lines = [
        "You are Bishop’s University Level-0 IT Support Assistant.",
        "Answer only for Bishop’s students/staff. Be concise.",
        "Use 3–6 numbered steps. If Level-0 steps fail, escalate to the Helpdesk at 819-822-9600 ext. 2273.",
        "",
        "Official topics and links (use only these):",
    ]
    for _, r in df.iterrows():
        intent = str(r.get("intent_name", "")).strip()
        link = (str(r.get("primary_link", "")).strip() or str(r.get("secondary_link", "")).strip())
        if intent and link:
            lines.append(f"- {intent}: {link}")
    return "\n".join(lines)

def links(text: str) -> str:
    return re.sub(r"(https?://[^\s]+)", r"<a href='\1' target='_blank'>\1</a>", text)

def clear_chat():
    st.session_state["messages"] = [{"role": "system", "content": st.session_state.get("system_prompt_text", SYSTEM_PROMPT_FALLBACK)}]
    st.session_state["latency_map"] = {}
    st.session_state["pending_clarify"] = None
    st.session_state["clarify_origin"] = None
    st.session_state["intent_lookup"] = None
    st.rerun()

# ---------------- init state ----------------
gt_df = load_gt()
st.session_state["system_prompt_text"] = build_system_prompt_from_gt(gt_df)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": st.session_state["system_prompt_text"]}]
if "latency_map" not in st.session_state:
    st.session_state["latency_map"] = {}
if "pending_clarify" not in st.session_state:
    st.session_state["pending_clarify"] = None
if "clarify_origin" not in st.session_state:
    st.session_state["clarify_origin"] = None

# ---------------- LLM helpers ----------------
MODEL_NAME = "llama3-70b-8192"

def llm_write_answer(user_question: str, intent_name: str, steps: str, primary_link: str = "") -> str:
    sys = st.session_state["system_prompt_text"]
    context = (
        "Use the following steps verbatim as constraints (do not invent new links):\n"
        f"INTENT: {intent_name}\n"
        f"STEPS:\n{steps}\n"
        f"PRIMARY_LINK: {primary_link or 'N/A'}\n"
        "Format: 1–2 summary sentences, then 3–6 numbered steps. Add one-line fallback at the end."
    )
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=600,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f"User question: {user_question}\n\n{context}"},
        ],
    )
    return resp.choices[0].message.content

def llm_write_oos(user_question: str) -> str:
    sys = st.session_state["system_prompt_text"]
    prompt = (
        "The question is out of scope for BU IT support (Wi-Fi, Moodle, email, MyBU, accounts, printing). "
        "Politely say you can only help with BU IT issues and provide the Helpdesk number 819-822-9600 ext. 2273.\n"
        f"User: {user_question}"
    )
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=120,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content

# -------- Deterministic clarify rendering --------
def write_clarify_plain(user_question: str, options: list[dict]) -> str:
    lines = ["Your question may refer to multiple intents. Reply with a number to confirm:\n"]
    for i, o in enumerate(options, 1):
        name = o.get("intent_name", "")
        lines.append(f"{i}. {name}")
    return "\n".join(lines)

# ---------------- intent lookup for clarify pick ----------------
def _load_intent_lookup():
    if "intent_lookup" in st.session_state and st.session_state["intent_lookup"]:
        return st.session_state["intent_lookup"]

    lookup = {}
    docs_path = os.path.join("artifacts", "docs.json")
    if os.path.exists(docs_path):
        try:
            with open(docs_path, "r", encoding="utf-8") as f:
                docs = json.load(f)
            for d in docs:
                iid = d.get("intent_id")
                if iid is not None:
                    lookup[int(iid)] = d
        except Exception:
            lookup = {}

    if not lookup and os.path.exists("enhanced_ground_truth.csv"):
        try:
            df = pd.read_csv("enhanced_ground_truth.csv")
            for _, r in df.iterrows():
                iid = r.get("intent_id")
                if pd.isna(iid):
                    continue
                iid = int(iid)
                lookup[iid] = {
                    "intent_id": iid,
                    "intent_name": str(r.get("intent_name", "")),
                    "steps": str(r.get("steps", "")),
                    "primary_link": str(r.get("primary_link", "")),
                }
        except Exception:
            pass

    st.session_state["intent_lookup"] = lookup
    return lookup

def _resolve_intent_info(chosen: dict):
    if chosen.get("steps") or chosen.get("primary_link"):
        return chosen
    iid = chosen.get("intent_id") or chosen.get("id")
    if iid is None:
        return chosen
    info = _load_intent_lookup().get(int(iid), {}).copy()
    for k in ("intent_name", "intent_id"):
        if chosen.get(k):
            info[k] = chosen[k]
    return info

# ---------------- Core turn ----------------
def handle_user_query(user_input: str):
    ui = (user_input or "").strip()

    # numeric choice from clarify phase
    if st.session_state.get("pending_clarify"):
        m = re.fullmatch(r"\s*([1-9])\s*$", ui)
        if m:
            idx = int(m.group(1)) - 1
            options = st.session_state.get("pending_clarify") or []
            if 0 <= idx < len(options):
                chosen = options[idx]
                info = _resolve_intent_info(chosen)
                origin_q = st.session_state.get("clarify_origin") or ui

                st.session_state.messages.append({"role": "user", "content": user_input})
                reply = llm_write_answer(
                    origin_q,
                    info.get("intent_name", ""),
                    info.get("steps", ""),
                    info.get("primary_link", "")
                )
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.session_state["pending_clarify"] = None
                st.session_state["clarify_origin"] = None
                st.rerun()
                return
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Please reply with a valid number from the list."}
                )
                st.rerun()
                return
        # non-numeric falls through

    # normal path
    st.session_state.messages.append({"role": "user", "content": user_input})
    start = time.time()
    try:
        res = pipeline_answer(user_input)

        if res.get("type") == "clarify":
            options = res.get("options", [])
            st.session_state["pending_clarify"] = options
            st.session_state["clarify_origin"] = user_input
            reply = write_clarify_plain(user_input, options)  # <- deterministic

        elif res.get("type") == "answer":
            steps = res.get("steps", "")
            link = res.get("primary_link", "")
            reply = llm_write_answer(user_input, res.get("intent_name", ""), steps, link)

            alts = res.get("alternatives", [])
            if alts:
                alt_names = [str(a.get("intent_name", "")) for a in alts]
                alt_str = " / ".join(alt_names)
                reply = f"{reply}\n\nOther possible intents: {alt_str}"

        else:
            reply = llm_write_oos(user_input)

    except Exception:
        reply = ("I'm having some technical difficulties right now. "
                 "Please try again later or contact IT helpdesk: 819-822-9600 ext. 2273.")
    end = time.time()

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state["latency_map"][len(st.session_state.messages) - 1] = end - start
    st.rerun()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🔗 Quick Help")
    st.markdown(
        "**Need immediate help?**\n\n"
        "📞 **Call IT Helpdesk:** 819-822-9600 ext. 2273\n\n"
        "📍 **Visit in person:** Library Learning Commons, 1st Floor"
    )
    st.header("📚 Quick Links")
    st.markdown(
        "- [🎓 Moodle Login](https://moodle.ubishops.ca/login/index.php/)\n"
        "- [🔑 Reset Password](https://passwordreset.microsoftonline.com/)\n"
        "- [📋 Support Tickets](https://octopus.ubishops.ca/)"
    )
    st.header("❓ FAQ Shortcuts")
    if st.button("How do I connect to Wi-Fi?"):
        handle_user_query("How do I connect to Wi-Fi?")
    if st.button("How can I access my Webmail?"):
        handle_user_query("How can I access my Webmail?")
    if st.button("How do I reset my Moodle password?"):
        handle_user_query("How do I reset my Moodle password?")
    st.markdown("---")
    if st.button("🧹 Clear chat", use_container_width=True):
        clear_chat()

# ---------------- History ----------------
for idx, msg in enumerate(st.session_state.messages[1:], start=1):
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(links(msg["content"]), unsafe_allow_html=True)
            latency = st.session_state["latency_map"].get(idx)
            if latency is not None:
                st.caption(f"⏱️ Response time: {latency:.1f}s")

# ---------------- Chat input ----------------
user_prompt = st.chat_input("Type your question here...")
if user_prompt:
    handle_user_query(user_prompt)
