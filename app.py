import os
import time
import uuid
from datetime import datetime
from typing import Generator, List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import requests
from datasets import Dataset, load_dataset, Features, Value, Sequence
import json


# ============ Setup ============

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # optional for web search
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USER = os.getenv("HF_USER")
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME")

if not OPENAI_API_KEY or not HF_TOKEN or not HF_USER or not HF_DATASET_NAME:
    st.error("Please set OPENAI_API_KEY, HF_TOKEN, HF_USER, HF_DATASET_NAME in your .env or Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ============ Hugging Face Dataset ============

dataset_id = f"{HF_USER}/{HF_DATASET_NAME}"

SOURCES_FEATURE = Sequence({
    "title": Value("string"),
    "url": Value("string"),
    "content": Value("string"),
})

FEATURES = Features({
    "session_id": Value("string"),
    "role": Value("string"),
    "content": Value("string"),
    "ts": Value("string"),
    "sources_json": Value("string"),  # list of {title,url,content}
    "web_search": Value("bool"),
})

def _empty_dataset():
    return Dataset.from_dict(
        {
            "session_id": [],
            "role": [],
            "content": [],
            "ts": [],
            "sources_json": [],
            "web_search": [],
        },
        features=FEATURES,
    )

def _normalize_sources(sources: list | None) -> list:
    if not sources:
        return []
    norm = []
    for s in sources:
        norm.append({
            "title": str(s.get("title", ""))[:500],
            "url": str(s.get("url", ""))[:1000],
            # accept either 'content' or 'snippet' from tool output
            "content": str(s.get("content", s.get("snippet", "")))[:4000],
        })
    return norm

try:
    hf_dataset = load_dataset(dataset_id, split="train")
    # If the existing dataset doesn't have 'sources_json', recreate with correct FEATURES
    if "page_id" in hf_dataset.column_names:
        hf_dataset = hf_dataset.remove_columns("page_id")
        hf_dataset.push_to_hub(dataset_id, token=HF_TOKEN)
    if any(col not in hf_dataset.column_names for col in ["web_search"]):
        hf_dataset = _empty_dataset()
        hf_dataset.push_to_hub(dataset_id, token=HF_TOKEN)
except Exception:
    hf_dataset = _empty_dataset()
    hf_dataset.push_to_hub(dataset_id, token=HF_TOKEN)



def log_message_hf(session_id: str, role: str, content: str, sources: list | None = None, web_search=False):
    """
    Append ONE row to the dataset and push. IMPORTANT:
    - pass scalars (not lists) for each column
    - 'sources' must match FEATURES (a list of dicts)
    """
    global hf_dataset
    row = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "ts": datetime.utcnow().isoformat(),
        "sources_json": json.dumps(_normalize_sources(sources), ensure_ascii=False),
        "web_search": web_search
    }
    hf_dataset = hf_dataset.add_item(row)
    hf_dataset.push_to_hub(dataset_id, token=HF_TOKEN)

# ============ Streamlit Page Setup ============

st.set_page_config(page_title="ChatGPT-like Web App", page_icon="💬", layout="centered")
st.title("💬 ChatGPT-style Chat")


# ============ Session State ============

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are ChatGPT: a friendly, concise, markdown-savvy assistant. "
                "Use conversational tone, structure answers with short paragraphs and lists when helpful. "
                "Cite sources only if given in a 'web_results' tool message. "
                "Do not reveal chain-of-thought. Be helpful and practical."
            ),
        }
    ]

if "title_set" not in st.session_state:
    st.session_state.title_set = False

# ============ Sidebar Controls ============

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model = st.selectbox(
        "Model",
        ["gpt-5"],
        index=0
    )

    if st.button("🆕 New chat"):
        st.session_state.clear()
        st.rerun()



# ============== Helper =====================
def sanitize_messages_for_responses(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Strip non-supported keys (e.g., 'sources') and coerce content to str."""
        clean: List[Dict[str, str]] = []
        for m in messages:
            role = m.get("role")
            if role not in ("system", "user", "assistant"):
                continue
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            clean.append({"role": role, "content": content})
        return clean


def extract_sources_from_raw_response_json(raw_json_str: str) -> List[Dict[str, str]]:
    """Parse all URLs from final response JSON (annotations + any sources arrays)."""
    if not raw_json_str:
        return []
    try:
        data = json.loads(raw_json_str)
    except Exception:
        return []

    collected: List[Dict[str, str]] = []

    # 1) Official 'sources' top-level (if the model returns it)
    top_sources = data.get("sources")
    if isinstance(top_sources, list):
        for s in top_sources:
            if isinstance(s, dict):
                collected.append({
                    "title": (s.get("title") or s.get("name") or "").strip(),
                    "url": (s.get("url") or s.get("link") or "").strip(),
                    "snippet": (s.get("snippet") or s.get("content") or s.get("description") or "").strip(),
                })

    # 2) Message parts -> annotations -> url_citation (matches your raw payload)
    for out in data.get("output", []) or []:
        if not isinstance(out, dict):
            continue
        for part in out.get("content", []) or []:
            if not isinstance(part, dict):
                continue
            # a) explicit 'annotations'
            anns = part.get("annotations") or []
            for a in anns:
                if isinstance(a, dict) and a.get("type") == "url_citation":
                    collected.append({
                        "title": (a.get("title") or "").strip(),
                        "url": (a.get("url") or "").strip(),
                        "snippet": "",
                    })
            # b) some SDKs also add 'sources' on the part
            part_sources = part.get("sources")
            if isinstance(part_sources, list):
                for s in part_sources:
                    if isinstance(s, dict):
                        collected.append({
                            "title": (s.get("title") or s.get("name") or "").strip(),
                            "url": (s.get("url") or s.get("link") or "").strip(),
                            "snippet": (s.get("snippet") or s.get("content") or s.get("description") or "").strip(),
                        })

    # Dedup by URL or title, preserve order
    seen, deduped = set(), []
    for s in collected:
        key = s.get("url") or s.get("title")
        if key and key not in seen:
            seen.add(key)
            deduped.append(s)
    return deduped


# ============ Streaming Wrapper ============

def stream_openai_response_with_sources(
    messages_or_text,
    model: str,
    enable_web_search: bool = True,
    on_delta=None,
):
    tools = [{"type": "web_search"}] if enable_web_search else []

    text_parts: List[str] = []
    sources: List[Dict[str, str]] = []
    raw_final_json = None

    def _norm_sources(items):
        if not items:
            return
        for r in items:
            if not isinstance(r, dict):
                continue
            title = (r.get("title") or r.get("name") or "").strip()
            url = (r.get("url") or r.get("link") or "").strip()
            snippet = (r.get("snippet") or r.get("content") or r.get("description") or "").strip()
            if url or title or snippet:
                sources.append({"title": title, "url": url, "snippet": snippet})

    def _try_parse_json(payload):
        try:
            obj = json.loads(payload)
            if isinstance(obj, dict) and "results" in obj:
                _norm_sources(obj["results"])
            elif isinstance(obj, list):
                _norm_sources(obj)
        except Exception:
            pass

    with client.responses.stream(
        model=model,
        input=messages_or_text,
        tools=tools,
    ) as stream:
        for event in stream:
            et = getattr(event, "type", "")

            if et == "response.output_text.delta":
                text_parts.append(event.delta)
                if on_delta:
                    on_delta("".join(text_parts))

            elif et == "response.tool_call.output":
                out = getattr(event, "output", None)
                if isinstance(out, dict):
                    _norm_sources(out.get("results"))
                    if isinstance(out.get("data"), list):
                        _norm_sources(out["data"])
                elif isinstance(out, str):
                    _try_parse_json(out)

            elif et == "response.tool_call.output_json":
                outj = getattr(event, "output_json", None)
                if isinstance(outj, dict) and "results" in outj:
                    _norm_sources(outj["results"])
                elif isinstance(outj, list):
                    _norm_sources(outj)

            elif et == "response.citations.delta":
                cits = getattr(event, "citations", None)
                if isinstance(cits, list):
                    for c in cits:
                        url = (c.get("url") or "").strip()
                        title = (c.get("title") or "").strip()
                        if url or title:
                            sources.append({"title": title, "url": url, "snippet": ""})

            elif et == "response.error":
                msg = getattr(event, "error", None)
                if msg and getattr(msg, "message", ""):
                    text_parts.append(f"\n\n_Error: {msg.message}_")
                    if on_delta:
                        on_delta("".join(text_parts))

        # === FINAL RESPONSE: parse annotations (your raw JSON shows them here) ===
        try:
            final_resp = stream.get_final_response()
            raw_final_json = final_resp.model_dump_json(indent=2)

            sources_from_raw = extract_sources_from_raw_response_json(raw_final_json)
            if sources_from_raw:
                sources.extend(sources_from_raw)

            # 1) Official full sources, if present (some models expose this)
            all_sources = getattr(final_resp, "sources", None)
            if isinstance(all_sources, list):
                _norm_sources(all_sources)

            # 2) Parse message annotations (url_citation) — THIS MATCHES YOUR RAW JSON
            for out in getattr(final_resp, "output", []) or []:
                for part in getattr(out, "content", []) or []:
                    # part.annotations is a list of {type,url,title,...}
                    anns = getattr(part, "annotations", None)
                    if isinstance(anns, list):
                        for a in anns:
                            if isinstance(a, dict) and a.get("type") == "url_citation":
                                url = (a.get("url") or "").strip()
                                title = (a.get("title") or "").strip()
                                if url or title:
                                    sources.append({"title": title, "url": url, "snippet": ""})

                    # Some SDKs might also include part.sources (less common)
                    part_sources = getattr(part, "sources", None)
                    if isinstance(part_sources, list):
                        _norm_sources(part_sources)

        except Exception as e:
            dbg(f"⚠️ Error getting final response: {e}")

        stream.close()

    # Deduplicate by URL/title
    seen, deduped = set(), []
    for s in sources:
        key = s.get("url") or s.get("title")
        if key and key not in seen:
            seen.add(key)
            deduped.append(s)

    return "".join(text_parts), deduped



# ============ Render Existing Chat ============

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message("user" if msg["role"] == "user" else "assistant", avatar=avatar):
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for i, s in enumerate(msg["sources"], start=1):
                    st.markdown(f"{i}. [{s['title']}]({s['url']})")
        st.markdown(msg["content"])


# ============ Chat Input ============

user_input = st.chat_input("Type your message…")

if user_input:
    # 1) Show the user's message immediately in the UI
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # Also persist to session + HF right away
    st.session_state.messages.append({"role": "user", "content": user_input})
    log_message_hf(st.session_state.session_id, "user", user_input)

    if not st.session_state.title_set:
        st.session_state.title_set = True

    # Build messages for the API call
    # Build messages for the API call
    api_messages = list(st.session_state.messages)

    # (If you still add Tavily 'web_results' as a system msg, it’s fine — the sanitizer keeps role/content only.)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("_🌐 Searching and generating..._")

        # ✅ Pass sanitized messages (no 'sources' field) to Responses API
        sanitized = sanitize_messages_for_responses(api_messages)

        streamed_text, sources_from_tool = stream_openai_response_with_sources(
            messages_or_text=sanitized,
            model=model,
            enable_web_search=True,
            on_delta=lambda t: placeholder.markdown(t)
        )

        placeholder.markdown(streamed_text)

        if sources_from_tool:
            with st.expander("Sources"):
                for i, s in enumerate(sources_from_tool, start=1):
                    st.markdown(f"{i}. [{s.get('title') or 'Source'}]({s.get('url') or ''})")



    # 5) Save assistant message (with sources) to session + HF
    st.session_state.messages.append(
    {"role": "assistant", "content": streamed_text, "sources": sources_from_tool or None}
    )
    log_message_hf(
        st.session_state.session_id,
        "assistant",
        streamed_text,
        sources=sources_from_tool,
        web_search=bool(sources_from_tool) # will be normalized and stored as sources_json
    )

    # Tiny pause for UX smoothness, then re-render full history
    time.sleep(0.1)
    st.rerun()