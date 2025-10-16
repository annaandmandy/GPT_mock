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
})

def _empty_dataset():
    return Dataset.from_dict(
        {
            "session_id": [],
            "role": [],
            "content": [],
            "ts": [],
            "sources_json": [],
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
            "content": str(s.get("content", ""))[:4000],
        })
    return norm

try:
    hf_dataset = load_dataset(dataset_id, split="train")
    # If the existing dataset doesn't have 'sources_json', recreate with correct FEATURES
    if "page_id" in hf_dataset.column_names:
        hf_dataset = hf_dataset.remove_columns("page_id")
        hf_dataset.push_to_hub(dataset_id, token=HF_TOKEN)
    if any(col not in hf_dataset.column_names for col in ["sources_json"]):
        hf_dataset = _empty_dataset()
        hf_dataset.push_to_hub(dataset_id, token=HF_TOKEN)
except Exception:
    hf_dataset = _empty_dataset()
    hf_dataset.push_to_hub(dataset_id, token=HF_TOKEN)



def log_message_hf(session_id: str, role: str, content: str, sources: list | None = None):
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
    }
    hf_dataset = hf_dataset.add_item(row)
    hf_dataset.push_to_hub(dataset_id, token=HF_TOKEN)

# ============ Streamlit Page Setup ============

st.set_page_config(page_title="ChatGPT-like Web App", page_icon="💬", layout="centered")
st.title("💬 ChatGPT-style Chat")

# ============ Sidebar Controls ============

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model = st.selectbox(
        "Model",
        ["gpt-4.1", "gpt-4o", "gpt-4o-mini"],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.2, 0.7, 0.1)
    use_search = st.toggle("🔎 Enable real-time web search (Tavily)", value=True)
    if use_search and not TAVILY_API_KEY:
        st.info("Add `TAVILY_API_KEY` to use web search (optional).")
    if st.button("🆕 New chat"):
        st.session_state.clear()
        st.rerun()

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

# ============ Web Search Helper ============

def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    if not TAVILY_API_KEY:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "advanced",
                "include_domains": [],
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=25,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        simplified = []
        for r in results:
            simplified.append({
                "title": r.get("title") or "Source",
                "url": r.get("url") or "",
                "content": r.get("content") or r.get("snippet") or "",
            })
        return simplified
    except Exception as e:
        st.warning(f"Search error: {e}")
        return []

# ============ Streaming Wrapper ============

def create_streaming_chat_completion(client, model, messages, temperature):
    kwargs = {"model": model, "messages": messages, "stream": True}
    if model in {"gpt-4o", "gpt-4o-mini", "gpt-4.1"}:
        kwargs["temperature"] = temperature
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        if "Unsupported value: 'temperature'" in str(e) or "unsupported_value" in str(e):
            kwargs.pop("temperature", None)
            return client.chat.completions.create(**kwargs)
        raise

def stream_openai_response(messages, model, temperature):
    stream = create_streaming_chat_completion(client, model, messages, temperature)
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content

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

# ============ Helper ================
def stream_or_fallback(messages, model, temperature):
    """
    Yield tokens if streaming is allowed; otherwise fetch once and yield in chunks.
    Always returns an iterator (generator).
    """
    # Try true streaming
    try:
        stream = create_streaming_chat_completion(client, model, messages, temperature)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
        return
    except Exception:
        # Fall back to non-streaming
        pass

    # Non-streaming fallback
    try:
        full = client.chat.completions.create(model=model, messages=messages)
        text = full.choices[0].message.content or ""
        step = 24  # fake-typing chunk size
        for i in range(0, len(text), step):
            yield text[i:i+step]
    except Exception as e:
        # As a last resort, yield an error message once so the loop still iterates
        yield f"_Request failed: {e}_"


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
    api_messages = list(st.session_state.messages)

    # 2) (Optional) Show a loading indicator specifically for web search
    sources = []
    if use_search:
        with st.spinner("🔎 Searching the web…"):
            sources = tavily_search(user_input, max_results=6)
        if sources:
            tool_text = "web_results:\n" + "\n\n".join(
                [f"- {s['title']} ({s['url']})\n  {s['content'][:500]}" for s in sources]
            )
            api_messages.append({"role": "system", "content": tool_text})

    # 3) Create an assistant bubble and show a "loading / thinking" message,
    #    then live-update it as tokens stream in
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        # Show a clear loading line *before* tokens start arriving
        placeholder.markdown("_⏳ Generating answer…_")

        streamed_text_parts: List[str] = []

        def _stream_and_collect() -> Generator[str, None, None]:
            for token in stream_openai_response(api_messages, model=model, temperature=temperature):
                streamed_text_parts.append(token)
                # As tokens come in, update the same message
                placeholder.markdown("".join(streamed_text_parts))
                yield token

        # 4) While the model thinks/types, the placeholder keeps updating
        #    (this serves as your visible "loading" indicator too)
        for _ in _stream_and_collect():
            pass

        streamed_text = "".join(streamed_text_parts)

        # (Optional) show sources below the assistant reply
        if sources:
            with st.expander("Sources"):
                for i, s in enumerate(sources, start=1):
                    st.markdown(f"{i}. [{s['title']}]({s['url']})")

    # 5) Save assistant message (with sources) to session + HF
    st.session_state.messages.append(
        {"role": "assistant", "content": streamed_text, "sources": sources if sources else None}
    )
    log_message_hf(st.session_state.session_id, "assistant", streamed_text, sources=sources)

    # Tiny pause for UX smoothness, then re-render full history
    time.sleep(0.1)
    st.rerun()