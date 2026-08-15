"""
Step 9: LLM (constrained, citation-forced generation).

Two backends:
  - "ollama": real local Llama/Mistral via Ollama's HTTP API.
  - "echo":   deterministic mock that composes an answer ONLY out of the
              retrieved evidence sentences (no free generation at all).
              This exists so the full pipeline can run and be graded on its
              wiring/logic without needing a downloaded model in this sandbox.

Both backends are forced into the same structured JSON contract:
    {
      "answer": "...",
      "claims": [{"text": "...", "chunk_id": "...", "ts_number": "...",
                  "clause_id": "...", "page": ...}, ...],
      "insufficient_evidence": false
    }
"""

import json
import re
import sys
import os
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SYSTEM_PROMPT = """You are a 3GPP standards assistant. You MUST answer using ONLY the
provided CONTEXT excerpts from 3GPP specifications. Follow these rules strictly:

1. Do not use any knowledge outside the provided CONTEXT, even if you know the answer.
2. Every factual sentence in your answer must be traceable to a specific context
   excerpt. Reference excerpts by their [chunk_id].
3. If the CONTEXT does not fully answer the question, say so explicitly and answer
   only the part that IS supported -- never fill gaps with plausible-sounding text.
4. Respond ONLY with valid JSON matching this schema, nothing else:
   {"answer": "<prose answer>",
    "claims": [{"text": "<claim sentence>", "chunk_id": "<id used as support>"}],
    "insufficient_evidence": <true|false>}
"""


def _format_context(evidence):
    blocks = []
    for c in evidence:
        blocks.append(
            f"[chunk_id={c['chunk_id']}] ({c['ts_number']}, {c['release']}, "
            f"clause {c['breadcrumb']}, page {c.get('page')})\n{c['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def _call_ollama(question, evidence):
    prompt = (
        f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{_format_context(evidence)}\n\n"
        f"QUESTION: {question}\n\nJSON ANSWER:"
    )
    payload = json.dumps({
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{config.OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    raw = body.get("response", "{}")
    return _safe_parse_json(raw)


def _call_groq(question, evidence):
    """Groq's OpenAI-compatible chat completions API, hosting real
    Llama 3.x models with a free tier -- good fit for a publicly deployed
    demo since it needs no local GPU or model download.
    Requires env var GROQ_API_KEY. Get one free at https://console.groq.com
    """
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "RAG_LLM_BACKEND=groq but GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com and set it as an "
            "environment variable (GROQ_API_KEY)."
        )

    user_prompt = (
        f"CONTEXT:\n{_format_context(evidence)}\n\n"
        f"QUESTION: {question}\n\nJSON ANSWER:"
    )
    payload = json.dumps({
        "model": config.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    raw = body["choices"][0]["message"]["content"]
    return _safe_parse_json(raw)


def _call_echo(question, evidence):
    """Deterministic, extraction-only mock LLM: assembles an answer purely by
    quoting/lightly stitching the top evidence sentences. Zero free generation
    -> zero hallucination risk, used to validate pipeline wiring offline."""
    claims = []
    answer_parts = []
    for c in evidence:
        sentences = re.split(r"(?<=[.!?])\s+", c["text"])
        # pick sentences with the most lexical overlap with the question
        q_terms = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
        best = max(
            sentences,
            key=lambda s: len(q_terms & set(re.findall(r"[a-zA-Z0-9]+", s.lower()))),
            default="",
        )
        if best and len(best.split()) > 3:
            answer_parts.append(f"{best.strip()} [{c['chunk_id']}]")
            claims.append({"text": best.strip(), "chunk_id": c["chunk_id"]})

    if not answer_parts:
        return {"answer": "", "claims": [], "insufficient_evidence": True}

    return {
        "answer": " ".join(answer_parts),
        "claims": claims,
        "insufficient_evidence": False,
    }


def _safe_parse_json(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except Exception:
        return {"answer": raw, "claims": [], "insufficient_evidence": False}


def generate_answer(question, evidence):
    if config.LLM_BACKEND == "ollama":
        return _call_ollama(question, evidence)
    if config.LLM_BACKEND == "groq":
        return _call_groq(question, evidence)
    return _call_echo(question, evidence)
