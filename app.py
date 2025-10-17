import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
import requests
from datasets import Dataset, load_dataset, Features, Value, Sequence
import json

# ============ Setup ============

load_dotenv()

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SITE_URL = os.getenv("SITE_URL", "")     # optional: for OpenRouter ranking
SITE_NAME = os.getenv("SITE_NAME", "")   # optional: for OpenRouter ranking

# Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USER = os.getenv("HF_USER")
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME")

if not OPENROUTER_API_KEY or not HF_TOKEN or not HF_USER or not HF_DATASET_NAME:
    st.error("Please set OPENROUTER_API_KEY, HF_TOKEN, HF_USER, HF_DATASET_NAME in your .env or Secrets.")
    st.stop()

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
    "sources_json": Value("string"),  # keep schema; we won't populate for OpenRouter
    "web_search": Value("bool"),
    "model": Value("string")
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
            "model": [],
        },
        features=FEATURES,
    )

def _normalize_sources(sources: list | None) -> list:
    # We won't receive sources from OpenRouter by default, but keep function to preserve schema.
    if not sources:
        return []
    norm = []
    for s in sources:
        norm.append({
            "title": str(s.get("title", ""))[:500],
            "url": str(s.get("url", ""))[:1000],
            "content": str(s.get("content", s.get("snippet", "")))[:4000],
        })
    return norm

try:
    hf_dataset = load_dataset(dataset_id, split="train")
    # If legacy columns exist, normalize
    if "page_id" in hf_dataset.column_names:
        hf_dataset = hf_dataset.remove_columns("page_id")
        hf_dataset.push_to_hub(dataset_id, token=HF_TOKEN)
    if any(col not in hf_dataset.column_names for col in ["model"]):
        hf_dataset = _empty_dataset()
        hf_dataset.push_to_hub(dataset_id, token=HF_TOKEN)
except Exception:
    hf_dataset = _empty_dataset()
    hf_dataset.push_to_hub(dataset_id, token=HF_TOKEN)

def log_message_hf(session_id: str, role: str, content: str, sources: list | None = None, web_search: bool=False, model: str="openai/gpt-4o-mini"):
    """Append ONE row to the dataset and push."""
    global hf_dataset
    row = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "ts": datetime.utcnow().isoformat(),
        "sources_json": json.dumps(_normalize_sources(sources), ensure_ascii=False),
        "web_search": web_search,
        "model": model
    }
    hf_dataset = hf_dataset.add_item(row)
    hf_dataset.push_to_hub(dataset_id, token=HF_TOKEN)

# ============ Streamlit Page Setup ============

st.set_page_config(page_title="ChatGPT-like Web App", page_icon="💬", layout="centered")
st.title("💬 Chat (OpenRouter)")

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
                "When you use web search, include a citations list. "
                "Citations should be returned to the caller if available (title, url, snippet). "
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
        "Model (OpenRouter)",
        [
            "openai/gpt-4o-mini",
            "google/gemini-2.5-flash",
            "anthropic/claude-sonnet-4.5",
            "x-ai/grok-code-fast-1",
            "perplexity/sonar",
        ],
        index=0
    )
    enable_web_search = st.checkbox("Enable web search (:online / plugin)", value=True)
    web_max_results = st.number_input("Max web results", min_value=1, max_value=10, value=5, step=1)

    if st.button("🆕 New chat"):
        st.session_state.clear()
        st.rerun()


# ============== Helpers =====================

def sanitize_messages_for_openrouter(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Strip non-supported keys and return OpenAI-style chat messages."""
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

def _dedupe_sources(sources):
    seen = set()
    out = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        title = (s.get("title") or "").strip()
        url = (s.get("url") or "").strip()
        snippet = (s.get("snippet") or "").strip()
        key = url or title
        if key and key not in seen:
            seen.add(key)
            out.append({"title": title, "url": url, "snippet": snippet})
    return out

def fetch_openrouter_full_json(messages: List[Dict[str, str]], model_name: str, use_web: bool, max_results: int = 5):
    """Optional non-stream call to inspect the full JSON response for debugging."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if SITE_URL:
        headers["HTTP-Referer"] = SITE_URL
    if SITE_NAME:
        headers["X-Title"] = SITE_NAME

    payload = {
        "model": f"{model_name}:online" if use_web and ":online" not in model_name else model_name,
        "messages": messages,
        "stream": False,
    }
    if use_web:
        payload["plugins"] = [{"id": "web", "max_results": int(max_results)}]

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
    resp.raise_for_status()
    return resp.json()

def _collect_annotations_from_message(msg_obj, bucket):
    """
    OpenRouter providers (e.g., GPT-4o-mini, Gemini) can return:
      message.annotations: [
        { "type": "url_citation", "url_citation": { "title", "url", "content", ... } }, ...
      ]
    We extract (title, url, snippet).
    """
    if not isinstance(msg_obj, dict):
        return
    ann = msg_obj.get("annotations")
    if not isinstance(ann, list):
        return
    for a in ann:
        if not isinstance(a, dict):
            continue
        if a.get("type") != "url_citation":
            continue
        uc = a.get("url_citation") or {}
        title = (uc.get("title") or "").strip()
        url = (uc.get("url") or "").strip()
        snippet = (uc.get("content") or "").strip()  # map 'content' → snippet
        if title or url or snippet:
            bucket.append({"title": title, "url": url, "snippet": snippet})

def stream_openrouter_chat(
    messages: List[Dict[str, str]],
    model_name: str,
    use_web: bool,
    max_results: int = 5,
    on_raw=lambda _line: None,  # debug hook
):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if SITE_URL:
        headers["HTTP-Referer"] = SITE_URL
    if SITE_NAME:
        headers["X-Title"] = SITE_NAME

    payload = {
        "model": f"{model_name}:online" if use_web and ":online" not in model_name else model_name,
        "messages": messages,
        "stream": True,
    }
    if use_web:
        payload["plugins"] = [{"id": "web", "max_results": int(max_results)}]

    def _dedupe(sources):
        seen = set(); out = []
        for s in sources or []:
            if not isinstance(s, dict): 
                continue
            title = (s.get("title") or "").strip()
            url = (s.get("url") or "").strip()
            snippet = (s.get("snippet") or "").strip()
            key = url or title
            if key and key not in seen:
                seen.add(key)
                out.append({"title": title, "url": url, "snippet": snippet})
        return out

    def _maybe_collect_from_message(msg_obj, bucket):
        # metadata.citations (some providers)
        meta = (msg_obj or {}).get("metadata") or {}
        meta_cits = meta.get("citations")
        if isinstance(meta_cits, list):
            for c in meta_cits:
                if isinstance(c, dict):
                    bucket.append({
                        "title": (c.get("title") or "").strip(),
                        "url": (c.get("url") or "").strip(),
                        "snippet": (c.get("snippet") or "").strip(),
                    })
        # message.citations (some providers)
        direct_cits = (msg_obj or {}).get("citations")
        if isinstance(direct_cits, list):
            for c in direct_cits:
                if isinstance(c, dict):
                    bucket.append({
                        "title": (c.get("title") or "").strip(),
                        "url": (c.get("url") or "").strip(),
                        "snippet": (c.get("snippet") or "").strip(),
                    })
        # NEW: message.annotations[*].url_citation (your examples)
        _collect_annotations_from_message(msg_obj, bucket)

    with requests.post(url, headers=headers, data=json.dumps(payload), stream=True, timeout=600) as r:
        r.raise_for_status()
        final_text_parts: List[str] = []
        sources: List[Dict[str, str]] = []

        for raw in r.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            on_raw(raw)  # debug hook

            if not raw.startswith("data: "):
                continue
            data_str = raw[len("data: "):].strip()
            if data_str == "[DONE]":
                break

            try:
                obj = json.loads(data_str)

                # 1) Text delta
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    final_text_parts.append(delta)

                # 2) Top-level streaming citations (plugin 'web')
                if isinstance(obj.get("citations"), list):
                    for c in obj["citations"]:
                        if isinstance(c, dict):
                            sources.append({
                                "title": (c.get("title") or "").strip(),
                                "url": (c.get("url") or "").strip(),
                                "snippet": (c.get("snippet") or "").strip(),
                            })

                # 3) Some providers include partial/complete message objects — harvest there too
                msg_obj = obj.get("choices", [{}])[0].get("message") or {}
                if msg_obj:
                    _maybe_collect_from_message(msg_obj, sources)

                yield "".join(final_text_parts), _dedupe(sources)

            except Exception:
                continue

        # Final yield
        yield "".join(final_text_parts), _dedupe(sources)




# ============ Render Existing Chat ============

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message("user" if msg["role"] == "user" else "assistant", avatar=avatar):
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for i, s in enumerate(msg["sources"], start=1):
                    st.markdown(f"{i}. [{s.get('title') or 'Source'}]({s.get('url') or ''})")
        st.markdown(msg["content"])


# ============ Chat Input ============

user_input = st.chat_input("Type your message…")

if user_input:
    # 1) Show user's message immediately
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # Persist to history + HF
    st.session_state.messages.append({"role": "user", "content": user_input})
    log_message_hf(st.session_state.session_id, "user", user_input, model="user-prompt")

    if not st.session_state.title_set:
        st.session_state.title_set = True

    api_messages = list(st.session_state.messages)
    sanitized = sanitize_messages_for_openrouter(api_messages)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("_✨ Generating..._")

        # local buffer for this turn
        turn_raw_lines = []

        streamed_text = ""
        sources_from_tool = []

        try:
            for partial_text, partial_sources in stream_openrouter_chat(
                sanitize_messages_for_openrouter(st.session_state.messages),
                model_name=model,
                use_web=enable_web_search,
                max_results=web_max_results
            ):
                streamed_text = partial_text
                sources_from_tool = partial_sources or []
                placeholder.markdown(streamed_text if streamed_text else "_✨ Generating..._")
        except requests.HTTPError as e:
            streamed_text = f"**HTTP Error**: {e}"
            placeholder.markdown(streamed_text)
        except Exception as e:
            streamed_text = f"**Error**: {e}"
            placeholder.markdown(streamed_text)

        # Optional full JSON mirror
        full_json = None
        
        full_json = fetch_openrouter_full_json(
            sanitize_messages_for_openrouter(st.session_state.messages),
            model_name=model,
            use_web=enable_web_search,
            max_results=web_max_results,
        )
        # After you have: full_json = fetch_openrouter_full_json(...)
        choices = (full_json or {}).get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            meta = msg.get("metadata") or {}

            merged = (sources_from_tool or []).copy()

            # metadata.citations
            if isinstance(meta.get("citations"), list):
                for c in meta["citations"]:
                    if isinstance(c, dict):
                        merged.append({
                            "title": (c.get("title") or "").strip(),
                            "url": (c.get("url") or "").strip(),
                            "snippet": (c.get("snippet") or "").strip(),
                        })

            # message.citations
            if isinstance(msg.get("citations"), list):
                for c in msg["citations"]:
                    if isinstance(c, dict):
                        merged.append({
                            "title": (c.get("title") or "").strip(),
                            "url": (c.get("url") or "").strip(),
                            "snippet": (c.get("snippet") or "").strip(),
                        })

            # NEW: message.annotations[*].url_citation
            _collect_annotations_from_message(msg, merged)

            # de-dupe
            seen = set(); deduped = []
            for s in merged:
                k = (s.get("url") or s.get("title") or "").strip()
                if k and k not in seen:
                    seen.add(k); deduped.append(s)
            sources_from_tool = deduped

        # ⬇️ show Sources expander like before
        if sources_from_tool:
            with st.expander("Sources"):
                for i, s in enumerate(sources_from_tool, start=1):
                    title = s.get("title") or "Source"
                    url = s.get("url") or ""
                    st.markdown(f"{i}. [{title}]({url})")


    # Save assistant message (with sources) to session + HF
    st.session_state.messages.append(
        {"role": "assistant", "content": streamed_text, "sources": sources_from_tool or None}
    )
    log_message_hf(
        st.session_state.session_id,
        "assistant",
        streamed_text,
        sources=sources_from_tool,
        web_search=bool(sources_from_tool),  # ✅ mark True if we actually got citations
        model=model
    )


    time.sleep(0.1)
    st.rerun()

