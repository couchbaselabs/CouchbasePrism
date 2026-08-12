"""Thin OpenAI chat helpers.

Every LLM call in PRISM goes through here so the governance boundary is easy
to audit: the model plans, binds, proposes and narrates - it never performs
arithmetic that an approved dictionary entry governs. See
design/architecture.md §6.
"""
import json
import os
import time

import requests

from . import config, trace

CHAT_URL = "https://api.openai.com/v1/chat/completions"


def _post(body: dict, timeout: int) -> dict:
    started = time.perf_counter()
    try:
        resp = requests.post(
            CHAT_URL,
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json=body, timeout=timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        trace.add("llm_call", endpoint=CHAT_URL, request=body, response=content,
                  usage=result.get("usage"),
                  elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
        return result
    except Exception as exc:
        trace.add("llm_call", endpoint=CHAT_URL, request=body,
                  error=f"{type(exc).__name__}: {exc}",
                  elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
        raise


def chat_json(system: str, user: str, model: str = None, timeout: int = 90) -> dict:
    """Structured call. Returns the parsed JSON object the model produced."""
    body = {
        "model": model or config.OPENAI_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    return json.loads(_post(body, timeout)["choices"][0]["message"]["content"])


def chat_text(prompt: str, model: str = None, max_tokens: int = 500,
              timeout: int = 90) -> str:
    """Free-text call, used only for answer synthesis."""
    body = {
        "model": model or config.OPENAI_MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    return _post(body, timeout)["choices"][0]["message"]["content"]
