#!/usr/bin/env python3
"""talktocode.indexing.llm_enricher

Thin helper around the OpenAI ChatCompletion API that produces *semantic* metadata
for entities extracted by the AST pipeline.  Currently it implements an
`summarize_entity()` function that returns a one-sentence description of an entity
if none is available from docstrings/comments.

Design goals
------------
• **Cache everything** – we persist responses under `.talktocode_cache/llm_enrichment/` so
  re-indexing the same file costs $0.
• **Low latency & cost** – limit prompts to < 600 tokens and cap the
  response at ~120 tokens.  Use deterministic temperature=0.
• **Safe fallback** – if the OpenAI call fails, return "No summary available".
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI  # `openai` must be in requirements.txt already
from tenacity import retry, wait_random_exponential, stop_after_attempt

# ---------------------------------------------------------------------------
# Local cache setup
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(".talktocode_cache") / "llm_enrichment"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Minimal OpenAI client wrapper  (lazy init – avoids import storms in Streamlit)
# ---------------------------------------------------------------------------
_CLIENT: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI()
    return _CLIENT


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _cache_path(prompt: str) -> Path:
    """Deterministic cache filename for a given prompt string."""
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{h}.json"


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(2))
def _query_chat_completion(prompt: str) -> str:
    """Call OpenAI chat completions with a fixed, safe configuration."""
    response = _get_client().chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
        max_tokens=120,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_entity(entity: Dict[str, Any]) -> str:
    """Return a one-sentence summary for a code entity dict.

    Parameters
    ----------
    entity: dict
        The dictionary representation of a CodeEntity (output of ``to_dict()``)
    Returns
    -------
    str
        A concise, human-readable description.  Falls back to
        "No summary available" on error.
    """
    # Quick exit if description already present (respect manual docs)
    if entity.get("description") and entity["description"] != "No description available":
        return entity["description"]

    # Build the prompt – keep it compact
    code_snippet = entity.get("code_snippet", "")[:300]  # cap length
    prompt = (
        "You are a senior Python engineer. In ONE concise sentence (max 25 words) "
        "summarise what the following code entity does. If you cannot infer it, "
        "reply exactly: No summary available.\n\n"
        f"Name: {entity.get('name')}\n"
        f"Type: {entity.get('type')}\n"
        f"Existing docstring: {entity.get('docstring') or ''}\n"
        f"Code snippet:\n{code_snippet}"
    )

    cache_file = _cache_path(prompt)
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())['summary']
        except Exception:
            pass  # ignore cache corruption and regenerate

    try:
        summary = _query_chat_completion(prompt)
    except Exception as e:
        # Log once and fall back
        print(f"[llm_enricher] OpenAI call failed: {e}")
        summary = "No summary available"

    try:
        cache_file.write_text(json.dumps({"summary": summary}))
    except Exception:
        pass  # silently ignore write errors

    return summary 