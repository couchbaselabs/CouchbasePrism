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


def resolve_model(model, stage: str = None) -> str:
    """`model` may be a single id, or a {stage: id} mapping when the caller
    selects a different model per pipeline stage. None falls back to the
    configured model for that stage's role, so call sites never need to know
    which model they are using."""
    if isinstance(model, dict):
        return model.get(stage) or model.get("default") or config.model_for(stage)
    return model or config.model_for(stage)


# Parameters a given model rejects, learned from its own 400 rather than from a
# hardcoded model list. Model families disagree about these and the set changes
# with each release: gpt-5.x requires `max_completion_tokens` and gpt-5.5
# accepts only the default temperature, while gpt-4o and gpt-4.1 take both of
# the older forms. A version table here would be wrong within a release or two.
_REJECTED_PARAMS = {}


def _adapt(body: dict) -> dict:
    """Rewrite a request for what this model is known to accept."""
    rejected = _REJECTED_PARAMS.get(body.get("model"), set())
    if "temperature" in rejected:
        body.pop("temperature", None)
    if "max_tokens" in rejected and "max_tokens" in body:
        body["max_completion_tokens"] = body.pop("max_tokens")
    return body


def _learn_rejection(resp) -> str:
    """The parameter the API just refused, or None if the 400 was about
    something else and retrying would be pointless."""
    try:
        param = (resp.json().get("error") or {}).get("param")
    except ValueError:
        return None
    return param if param in ("temperature", "max_tokens") else None


def _post(body: dict, timeout: int, stage: str = None) -> dict:
    """`stage` names which pipeline step made the call - planner, binder,
    candidate, answer, judge. It is recorded on the trace event so the UI can
    label calls without inspecting prompt text; sniffing prompts for
    identifying phrases silently misclassified every call the moment a prompt
    was reworded. It also selects the model, via resolve_model.
    """
    started = time.perf_counter()
    body = _adapt(body)
    while True:
        try:
            resp = requests.post(
                CHAT_URL,
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json=body, timeout=timeout,
            )
            if resp.status_code == 400:
                param = _learn_rejection(resp)
                if param and param not in _REJECTED_PARAMS.setdefault(
                        body.get("model"), set()):
                    # Learn it once per model, then retry the same call. Every
                    # later call for this model is adapted before it is sent.
                    _REJECTED_PARAMS[body["model"]].add(param)
                    body = _adapt(dict(body))
                    continue
            resp.raise_for_status()
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            trace.add("llm_call", stage=stage, endpoint=CHAT_URL, request=body,
                      response=content, usage=result.get("usage"),
                      elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
            return result
        except Exception as exc:
            trace.add("llm_call", stage=stage, endpoint=CHAT_URL, request=body,
                      error=f"{type(exc).__name__}: {exc}",
                      elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
            raise


def chat_json(system: str, user: str, model=None, timeout: int = 90,
              stage: str = None) -> dict:
    """Structured call. Returns the parsed JSON object the model produced."""
    body = {
        "model": resolve_model(model, stage),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    return json.loads(_post(body, timeout, stage)["choices"][0]["message"]["content"])


def chat_text(prompt: str, model=None, max_tokens: int = 500,
              timeout: int = 90, stage: str = None) -> str:
    """Free-text call, used only for answer synthesis."""
    body = {
        "model": resolve_model(model, stage),
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    return _post(body, timeout, stage)["choices"][0]["message"]["content"]
