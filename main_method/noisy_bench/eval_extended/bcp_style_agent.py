"""BCP-CANONICAL deep-research agent (DeepSeek / MiMo backend, chat.completions).

Aligned to BrowseComp-Plus's leaderboard configuration so our noisy-bench numbers
are comparable to BCP. Verified against BCP source (search_agent/oss_client.py,
openai_client.py, prompts.py, searcher/searchers/base.py) on 2026-05-29:

  - Prompt = BCP's `QUERY_TEMPLATE_NO_GET_DOCUMENT` (search-only, verbatim).
  - ONE tool: `search` (name + `query` param match openai_client.py); description
    is BCP's `search_description(k)` verbatim. NO get_document (BCP default).
  - max_iterations = 100 (BCP default), NO max_searches cap (BCP has none).
  - snippet = 512 tokens (BCP default `--snippet-max-tokens`), truncated with the
    Qwen3 tokenizer (BCP uses Qwen/Qwen3-0.6B; the whole Qwen3 family shares one
    tokenizer, so the local Qwen3-Embedding tokenizer gives identical token counts
    and needs no download).
  - max_tokens (generation) = 10000 (BCP `--max-tokens` default).
  - k = 5 (BCP default).
  - Answer extraction = BCP semantics: a model turn with no tool call IS the final
    answer; loop exhaustion (100 rounds) -> status "incomplete". No forced-answer
    salvage (BCP does none; forcing was what previously caused the DSML leaks).

The ONLY unavoidable divergence is the LLM API surface: DeepSeek/MiMo speak Chat
Completions, not the OpenAI Responses API that BCP's openai_client/oss_client use
(no BCP client uses Responses for these providers either). `reasoning_content` is
echoed back each turn because the thinking models require it; that is an API-surface
necessity, not a behavioural change.

Emits BCP-format per-query JSON for the existing grader (evaluate_noisy.py):
    {query_id, model, condition, tool_call_counts, retrieved_docids,
     status, result:[{type:"tool_call",...}, {type:"output_text","output":...}]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from noisy_bench.eval_extended.evidence_gate import (
    GENERATED_GATE_VERSION,
    EvidenceCompletionGate,
    GateAction,
    GateController,
    GateDecision,
    GateState,
    continuation_message,
    insufficient_response,
    load_obligations,
    verified_response,
)


# transformers 4.57 treats Hugging Face ``kernels`` as optional, but the
# currently installed kernels 0.16/huggingface-hub 0.36 pair fails during
# import on Python 3.13 before transformers can catch it as an ImportError.
# This runner only needs a tokenizer; it never uses optional hub kernels.
if sys.version_info >= (3, 13):
    sys.modules.setdefault("kernels", None)

# BCP's exact canonical instruction template (search-only variant, verbatim from
# BrowseComp-Plus/search_agent/prompts.py :: QUERY_TEMPLATE_NO_GET_DOCUMENT).
QUERY_TEMPLATE_NO_GET_DOCUMENT = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()


# --- E3 verification-intervention variants (ablation only). "none" = BCP-canonical
#     (empty suffix -> the prompt is byte-identical to the baseline, so canonical
#     results are unaffected). "verify"/"enforce" append a policy paragraph. ---
INSTRUCTION_VARIANTS = {
    "none": "",
    "generic": (
        # GENERIC verification/diligence nudge — does NOT mention summaries or tell the model to distrust "
        # any document class, so it does not leak the attack. Targets premature stopping only.
        "Before you give your Exact Answer, verify it against the evidence: make sure the specific documents "
        "you retrieved actually support it, and keep searching to confirm rather than stopping at the first "
        "plausible answer."
    ),
    "verify": (
        "Note: some retrieved documents may be high-level summaries or briefs that directly state an "
        "answer. Do not treat a directly-stated answer from such a summary as authoritative on its own. "
        "Before you finalize, cross-check any directly-stated answer against the underlying record-level "
        "evidence you can retrieve; if a summary conflicts with the record-level entries, rely on the "
        "record-level evidence."
    ),
    "enforce": (
        "Verification policy (required): Do not finalize an answer on the basis of a single summary document "
        "that directly states it. Before giving your 'Exact Answer', you must retrieve and cite at least two "
        "independent record-level documents (individual records or entries, not high-level summaries) whose "
        "contents together support your answer. If any summary conflicts with the record-level evidence, "
        "follow the record-level evidence."
    ),
}


def _search_description(k: int) -> str:
    """Verbatim BCP `BaseSearcher.search_description(k)` (searcher/searchers/base.py)."""
    return (
        f"Perform a search on a knowledge source. Returns top-{k} hits with docid, "
        "score, and snippet. The snippet contains the document's contents (may be "
        "truncated based on token limits)."
    )


def _tools(k: int) -> List[Dict[str, Any]]:
    """Single `search` tool — name/param/description aligned to BCP openai_client.py."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": _search_description(k),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to search the local knowledge base for relevant information",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]


def _serialize_tool_call(tc) -> Dict[str, Any]:
    """Preserve provider extensions required in subsequent tool-call turns."""
    serialized = {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }
    raw = tc.model_dump(exclude_none=True) if hasattr(tc, "model_dump") else {}
    extra_content = raw.get("extra_content")
    if extra_content is None:
        extra_content = getattr(tc, "extra_content", None)
    if extra_content is None:
        extra_content = (getattr(tc, "model_extra", None) or {}).get("extra_content")
    if extra_content is not None:
        serialized["extra_content"] = extra_content
    return serialized


# --- snippet tokenizer (BCP truncates snippets to N tokens; Qwen3 family shares
#     one tokenizer, so the local Qwen3-Embedding tokenizer == Qwen/Qwen3-0.6B for
#     token counts, and needs no download). Lazy-loaded once per process. ---
_SNIPPET_TOKENIZER = None
_SNIPPET_TOKENIZER_PATH = os.environ.get("QWEN_TOKENIZER", "Qwen/Qwen3-Embedding-8B")


def _get_snippet_tokenizer(path: str):
    global _SNIPPET_TOKENIZER
    if _SNIPPET_TOKENIZER is None:
        from transformers import AutoTokenizer
        _SNIPPET_TOKENIZER = AutoTokenizer.from_pretrained(path, revision=os.environ.get("EMBEDDING_REVISION"))
    return _SNIPPET_TOKENIZER


def _truncate_to_tokens(text: str, max_tokens: int, tokenizer) -> str:
    """BCP snippet truncation (oss_client.SearchToolHandler._search), verbatim logic."""
    if not text or not max_tokens or max_tokens <= 0:
        return text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    return text


def _doc_contents(searcher, docid: str) -> Optional[str]:
    """Full document text from the Lucene index (stored as {id, contents})."""
    try:
        doc = searcher.doc(str(docid))
        if doc is None:
            return None
        raw = doc.raw()
        if not raw:
            return None
        try:
            return json.loads(raw).get("contents", "")
        except Exception:
            return raw
    except Exception:
        return None


class _Hit:
    """Minimal duck-type for LuceneSearcher hits used by _search()."""
    __slots__ = ("docid", "score")

    def __init__(self, docid: str, score: float):
        self.docid = str(docid)
        self.score = float(score)


class DenseSearcher:
    """FAISS dense retrieval over a Qwen3-Embedding-8B index, with full-text
    fallback to a sibling pyserini Lucene index (the dense pickle has no raw
    documents, so snippet text is fetched from Lucene by docid)."""

    def __init__(self, dense_index_dir: str, encode_server_url: str,
                 docs_lucene_index: str, request_timeout: float = 30.0):
        import faiss
        import pickle
        import requests
        from pyserini.search.lucene import LuceneSearcher

        self.encode_server_url = encode_server_url.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False
        self.request_timeout = request_timeout

        with open(f"{dense_index_dir}/corpus.pkl", "rb") as f:
            reps, lookup = pickle.load(f)
        self.lookup = [str(x) for x in lookup]
        # IndexFlatIP requires normalized vectors for cosine semantics. Our build
        # script already normalizes; assert to catch a mis-built index early.
        norms = np.linalg.norm(reps[:3], axis=1)
        assert (np.abs(norms - 1.0) < 1e-2).all(), \
            f"dense index reps not L2-normalized: {norms}"
        self.index = faiss.IndexFlatIP(reps.shape[1])
        self.index.add(reps)
        self._docs = LuceneSearcher(str(docs_lucene_index))

        # Sanity ping
        r = self.session.get(f"{self.encode_server_url}/health",
                             timeout=self.request_timeout)
        r.raise_for_status()
        srv = r.json()
        assert srv.get("dim") == reps.shape[1], (
            f"server dim {srv.get('dim')} != index dim {reps.shape[1]}")

    def _encode_query(self, query: str) -> np.ndarray:
        r = self.session.post(f"{self.encode_server_url}/encode",
                              json={"query": query}, timeout=self.request_timeout)
        r.raise_for_status()
        return np.asarray(r.json()["vector"], dtype="float32")[None, :]

    def search(self, query: str, k: int = 10) -> List[_Hit]:
        q = self._encode_query(query)
        scores, idxs = self.index.search(q, k)
        return [_Hit(self.lookup[i], s) for s, i in zip(scores[0], idxs[0]) if i >= 0]

    def doc(self, docid: str):
        """Delegate to the Lucene side so _doc_contents works unchanged."""
        return self._docs.doc(str(docid))


class ExcludingSearcher:
    """Read-only search wrapper used by missing-evidence control conditions."""

    def __init__(self, searcher, excluded_docids):
        self.searcher = searcher
        self.excluded_docids = {str(x) for x in excluded_docids}

    def search(self, query: str, k: int = 10):
        hits = self.searcher.search(query, k=k + len(self.excluded_docids))
        return [h for h in hits if str(h.docid) not in self.excluded_docids][:k]

    def doc(self, docid: str):
        return self.searcher.doc(str(docid))


def _search(searcher, query: str, k: int, snippet_max_tokens: int, tokenizer) -> List[Dict[str, Any]]:
    """One `search` call: top-k hits, each snippet = full doc text truncated to
    `snippet_max_tokens` tokens (BCP SearchToolHandler._search semantics)."""
    hits = searcher.search(query, k=k)
    out = []
    for h in hits:
        full = _doc_contents(searcher, h.docid) or ""
        out.append({"docid": h.docid, "score": round(float(h.score), 4),
                    "snippet": _truncate_to_tokens(full, snippet_max_tokens, tokenizer)})
    return out


def run_one(
    *,
    query_id: str,
    query: str,
    condition: str,
    searcher,
    client,
    model: str,
    max_iterations: int = 100,
    k: int = 5,
    snippet_max_tokens: int = 512,
    snippet_tokenizer_path: str = _SNIPPET_TOKENIZER_PATH,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 10000,
    instruction_variant: str = "none",
    evidence_gate_mode: str = "none",
    gate_model: Optional[str] = None,
    gate_obligations: Optional[List[Dict[str, Any]]] = None,
    gate_max_rejections: int = 4,
    gate_stagnation_limit: int = 2,
    gate_corruption_budget: int = 1,
) -> Dict[str, Any]:
    """Run one BCP-canonical ReAct episode; return a BCP-format result dict.

    Loop semantics mirror BCP (oss_client.run_conversation_with_tools): up to
    `max_iterations` tool-call rounds; a model turn with no tool call is the final
    answer (status "completed"); exhausting the rounds -> status "incomplete".
    No max_searches cap and no forced-answer salvage (BCP has neither).
    """
    tokenizer = _get_snippet_tokenizer(snippet_tokenizer_path)
    tools = _tools(k)
    _content = QUERY_TEMPLATE_NO_GET_DOCUMENT.format(Question=query)
    _suffix = INSTRUCTION_VARIANTS.get(instruction_variant, "")
    if _suffix:                                  # variant="none" -> byte-identical to BCP baseline
        _content = _content + "\n\n" + _suffix
    messages = [{"role": "user", "content": _content}]
    result_log: List[Dict[str, Any]] = []
    retrieved: List[str] = []
    evidence_by_docid: Dict[str, str] = {}
    tool_counts = {"search": 0}
    status = "incomplete"
    final_text = ""
    usage_tot = {"prompt": 0, "completion": 0, "cached": 0}
    nudges = 0  # thinking models (Gemini) sometimes return empty content + no tool call
    gate = None
    gate_controller = None
    gate_log: List[Dict[str, Any]] = []
    if evidence_gate_mode != "none":
        obligations = load_obligations(gate_obligations) if gate_obligations else None
        gate = EvidenceCompletionGate(
            client=client,
            model=gate_model or model,
            mode=evidence_gate_mode,
            obligations=obligations,
        )
        gate.prepare(query)
        gate_controller = GateController(
            max_rejections=gate_max_rejections,
            stagnation_limit=gate_stagnation_limit,
        )

    try:
        for _ in range(max_iterations):
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=tools, tool_choice="auto",
                temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            )
            try:
                u = resp.usage
                usage_tot["prompt"] += u.prompt_tokens or 0
                usage_tot["completion"] += u.completion_tokens or 0
                ptd = getattr(u, "prompt_tokens_details", None)
                if ptd:
                    usage_tot["cached"] += getattr(ptd, "cached_tokens", 0) or 0
            except Exception:
                pass
            msg = resp.choices[0].message
            # Thinking models (deepseek-v4 / mimo): reasoning_content MUST be echoed
            # back next turn (API-surface necessity, not a behaviour change).
            reasoning = getattr(msg, "reasoning_content", None)

            tc_serial = None
            if msg.tool_calls:
                tc_serial = [_serialize_tool_call(tc) for tc in msg.tool_calls]
            am = {"role": "assistant", "content": msg.content or ""}
            if tc_serial:
                am["tool_calls"] = tc_serial
            if reasoning:
                am["reasoning_content"] = reasoning
            messages.append(am)

            # No tool call -> this turn is the final answer (BCP: `if not
            # function_calls: return ... "completed"`). Take content verbatim.
            if not msg.tool_calls:
                candidate_text = (msg.content or "").strip()
                if not candidate_text and reasoning:        # thinking model returned empty content
                    candidate_text = reasoning.strip()
                if not candidate_text and nudges < 2:        # nudge it to actually answer
                    nudges += 1
                    messages.append({"role": "user", "content":
                        "You have not provided a final answer. Using the search results above, "
                        "give your final answer now in the form 'Exact Answer: <answer>' with citations."})
                    continue

                if gate is not None and gate_controller is not None:
                    gate_usage_before = dict(getattr(gate, "usage_total", {}))
                    try:
                        decision = gate.review(
                            question=query,
                            candidate_answer=candidate_text,
                            evidence=[
                                {"docid": docid, "text": text}
                                for docid, text in evidence_by_docid.items()
                            ],
                            search_attempts=tool_counts["search"],
                            prior_rejections=gate_controller.rejections,
                        )
                    except Exception as exc:
                        decision = GateDecision(
                            state=GateState.INCOMPLETE,
                            missing=["The independent gate could not complete its audit"],
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                    gate_usage_after = getattr(gate, "usage_total", {})
                    for key in usage_tot:
                        usage_tot[key] += (
                            gate_usage_after.get(key, 0) - gate_usage_before.get(key, 0)
                        )
                    relevant_progress = gate.progress_count(evidence_by_docid, decision)
                    action = gate_controller.decide(decision, relevant_progress)
                    gate_record = {
                        "state": decision.state.value,
                        "action": action.value,
                        "reconstructed_answer": decision.reconstructed_answer,
                        "supporting_docids": decision.supporting_docids,
                        "missing": decision.missing,
                        "queries": decision.queries,
                        "reason": decision.reason,
                        "obligation_support": decision.obligation_support,
                        "n_unique_docs": len(evidence_by_docid),
                        "n_progress_docs": relevant_progress,
                        "candidate_answer": candidate_text,
                    }
                    if evidence_gate_mode == "sourcecut":
                        gate_record["sourcecut_audit"] = gate.last_audit
                    gate_log.append(gate_record)
                    if action == GateAction.ACCEPT:
                        final_text = (
                            verified_response(decision)
                            if evidence_gate_mode in {"generated", "reconciliation", "sourcecut"}
                            else candidate_text
                        )
                        status = "completed"
                        break
                    if action == GateAction.ABSTAIN:
                        if decision.state != GateState.INSUFFICIENT:
                            prior_reason = decision.reason
                            decision.state = GateState.INSUFFICIENT
                            decision.reason = (
                                "Search ended because the evidence gate reached its bounded "
                                "retry or evidence-stagnation limit."
                                + (f" Last audit: {prior_reason}" if prior_reason else "")
                            )
                        final_text = insufficient_response(decision)
                        status = "completed"
                        break
                    messages.append({"role": "user", "content": continuation_message(decision)})
                    continue

                final_text = candidate_text
                status = "completed"
                break

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                if name == "search":
                    q = str(args.get("query", "")).strip()
                    hits = _search(searcher, q, k, snippet_max_tokens, tokenizer) if q else []
                    retrieved.extend(h["docid"] for h in hits)
                    for hit in hits:
                        evidence_by_docid.setdefault(str(hit["docid"]), hit.get("snippet", ""))
                    tool_counts["search"] += 1
                    payload = hits
                    result_log.append({"type": "tool_call", "tool_name": "search",
                                       "arguments": tc.function.arguments,
                                       "output": json.dumps(hits, ensure_ascii=False)[:12000]})
                else:
                    payload = {"error": f"unknown tool {name}"}
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": name,
                                 "content": json.dumps(payload, ensure_ascii=False)})
    except Exception as e:
        status = f"error:{type(e).__name__}"

    result_log.append({"type": "output_text", "output": final_text})
    return {
        "query_id": query_id, "model": model, "condition": condition,
        "tool_call_counts": tool_counts,
        "retrieved_docids": retrieved,
        "status": status,
        "usage": usage_tot,
        "evidence_gate": {
            "version": (
                "sourcecut_v1" if evidence_gate_mode == "sourcecut"
                else GENERATED_GATE_VERSION if evidence_gate_mode == "generated"
                else None
            ),
            "mode": evidence_gate_mode,
            "model": gate_model or model if evidence_gate_mode != "none" else None,
            "obligation_source": (
                "question_only"
                if evidence_gate_mode == "generated"
                else "provided_diagnostic" if evidence_gate_mode == "reconciliation" else None
            ),
            "draft_obligations": [o.__dict__ for o in gate.draft_obligations] if gate else [],
            "obligations": [o.__dict__ for o in gate.obligations] if gate else [],
            "audits": gate_log,
        },
        "result": result_log,
    }


def run_one_gemini_native(
    *,
    query_id: str,
    query: str,
    condition: str,
    searcher,
    api_key: str,
    base_url: str,
    model: str,
    max_iterations: int = 100,
    k: int = 5,
    snippet_max_tokens: int = 512,
    snippet_tokenizer_path: str = _SNIPPET_TOKENIZER_PATH,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 32000,
    instruction_variant: str = "none",
) -> Dict[str, Any]:
    """Gemini 3.x native `generateContent` path. Gemini 3-generation models need the
    per-turn `thoughtSignature` echoed back to resume reasoning across tool calls; the
    OpenAI-compatible /chat/completions relays DROP it (they only surface reasoning_content
    text), which silently degrades multi-step tool use into early-commit/deference. The
    native REST endpoint returns thoughtSignature on the functionCall part, and we preserve
    the whole model turn (parts incl. thoughtSignature) in history. Same result dict as
    run_one(); reuses _search so retrieval is byte-identical to the OpenAI path."""
    tokenizer = _get_snippet_tokenizer(snippet_tokenizer_path)
    host = base_url.rstrip("/")
    if host.endswith("/v1"):
        host = host[:-3]
    url = f"{host}/v1beta/models/{model}:generateContent"
    # BCP-canonical name/description/params, identical to the OpenAI path (run_one uses
    # _tools(k)); Gemini's function_declaration is exactly that function object.
    fdecl = [_tools(k)[0]["function"]]
    _content = QUERY_TEMPLATE_NO_GET_DOCUMENT.format(Question=query)
    _suffix = INSTRUCTION_VARIANTS.get(instruction_variant, "")
    if _suffix:
        _content = _content + "\n\n" + _suffix
    contents = [{"role": "user", "parts": [{"text": _content}]}]
    result_log: List[Dict[str, Any]] = []
    retrieved: List[str] = []
    tool_counts = {"search": 0}
    status = "incomplete"
    final_text = ""
    nudges = 0  # empty-answer re-prompts, mirroring run_one
    usage_tot = {"prompt": 0, "completion": 0, "cached": 0}

    headers = {"Content-Type": "application/json",
               "x-goog-api-key": api_key, "Authorization": f"Bearer {api_key}"}

    def _post(body):
        data = json.dumps(body).encode()
        backoff = 0
        for _ in range(40):  # transient geo-block (400, pre-inference), 429/5xx, timeout
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=180) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                msg = e.read().decode()[:300]
                if "location is not supported" in msg:
                    time.sleep(0.5)
                    continue
                if e.code == 429 or e.code >= 500:  # rate-limit / server: back off & retry
                    backoff += 1
                    time.sleep(min(2 ** backoff, 30))
                    continue
                raise RuntimeError(f"HTTP {e.code}: {msg}")
            except Exception as e:
                if "timed out" in str(e).lower():
                    continue
                raise
        raise RuntimeError("relay: retries exhausted")

    try:
        for _ in range(max_iterations):
            r = _post({
                "contents": contents,
                "tools": [{"function_declarations": fdecl}],
                "generationConfig": {"maxOutputTokens": max_tokens,
                                     "temperature": temperature, "topP": top_p},
            })
            um = r.get("usageMetadata", {}) or {}
            usage_tot["prompt"] += um.get("promptTokenCount", 0) or 0
            usage_tot["completion"] += (um.get("candidatesTokenCount", 0) or 0) + (um.get("thoughtsTokenCount", 0) or 0)
            usage_tot["cached"] += um.get("cachedContentTokenCount", 0) or 0
            cands = r.get("candidates") or []
            if not cands:
                status = "error:no_candidates"
                break
            parts = (cands[0].get("content") or {}).get("parts") or []
            fcs = [p for p in parts if "functionCall" in p]
            if fcs:
                contents.append({"role": "model", "parts": parts})  # PRESERVE thoughtSignature
                resp_parts = []  # all functionResponses in ONE user turn (Gemini requires this)
                for p in fcs:
                    fc = p["functionCall"]
                    if fc.get("name") == "search":
                        q = str((fc.get("args") or {}).get("query", "")).strip()
                        hits = _search(searcher, q, k, snippet_max_tokens, tokenizer) if q else []
                        retrieved.extend(h["docid"] for h in hits)
                        tool_counts["search"] += 1
                        result_log.append({"type": "tool_call", "tool_name": "search",
                                           "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False),
                                           "output": json.dumps(hits, ensure_ascii=False)[:12000]})
                        resp_parts.append({"functionResponse":
                            {"name": "search", "response": {"result": hits}}})
                    else:  # unknown-tool guard (mirror run_one)
                        resp_parts.append({"functionResponse": {"name": fc.get("name") or "unknown",
                            "response": {"error": f"unknown tool {fc.get('name')}"}}})
                contents.append({"role": "user", "parts": resp_parts})
            else:
                final_text = " ".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
                if not final_text and nudges < 2:  # empty answer (e.g. truncation) -> re-prompt
                    nudges += 1
                    contents.append({"role": "model", "parts": parts or [{"text": ""}]})
                    contents.append({"role": "user", "parts": [{"text":
                        "You have not provided a final answer. Using the search results above, "
                        "give your final answer now in the form 'Exact Answer: <answer>' with citations."}]})
                    continue
                status = "completed"
                break
    except Exception as e:
        status = f"error:{type(e).__name__}"

    result_log.append({"type": "output_text", "output": final_text})
    return {
        "query_id": query_id, "model": model, "condition": condition,
        "tool_call_counts": tool_counts,
        "retrieved_docids": retrieved,
        "status": status,
        "usage": usage_tot,
        "result": result_log,
    }


def main():
    ap = argparse.ArgumentParser(description="BCP-canonical DeepSeek/MiMo deep-research agent.")
    ap.add_argument("--queries", required=True, type=Path, help="JSONL with query_id, query, answer")
    ap.add_argument("--bm25_index", type=Path,
                    help="Pyserini Lucene index dir. Required for BM25 retrieval and as the "
                         "full-text source for search snippets on the dense path.")
    ap.add_argument("--dense_index", type=Path, default=None,
                    help="Dense (FAISS) index dir produced by build_dense_index_qwen3_8b.py. "
                         "When set, search() goes through DenseSearcher + --encode_server.")
    ap.add_argument("--encode_server", default="http://127.0.0.1:8765",
                    help="URL of the dense_server.py query-encoding service")
    ap.add_argument("--condition", required=True, help="clean / K1 / K3 / K5 (label only)")
    ap.add_argument("--output_dir", required=True, type=Path, help="dir for per-query BCP-format JSON")
    ap.add_argument("--model", default="deepseek-v4-flash")
    # BCP-canonical defaults (search_agent/oss_client.py + openai_client.py):
    ap.add_argument("--max_iterations", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--snippet_max_tokens", type=int, default=512)
    ap.add_argument("--max_tokens", type=int, default=10000)
    ap.add_argument("--instruction_variant", default="none", choices=list(INSTRUCTION_VARIANTS),
                    help="E3 ablation only. 'none' = BCP-canonical prompt (default, byte-identical). "
                         "'verify'/'enforce' append a verification-policy paragraph.")
    ap.add_argument("--evidence_gate", default="none",
                    choices=["none", "global", "generated", "reconciliation"],
                    help="Optional final-answer gate. 'global' is a conventional sufficiency judge; "
                         "'generated' plans atomic claims from the question alone; 'reconciliation' "
                         "uses frozen route obligations as a diagnostic upper bound.")
    ap.add_argument("--gate_model", default=None,
                    help="Verifier model (defaults to --model).")
    ap.add_argument("--gate_obligations", type=Path, default=None,
                    help="Optional JSON with an obligations list, or a query-id keyed map. "
                         "This is forbidden in generated mode to prevent oracle leakage.")
    ap.add_argument("--gate_max_rejections", type=int, default=4)
    ap.add_argument("--gate_stagnation_limit", type=int, default=2)
    ap.add_argument("--gate_corruption_budget", type=int, default=1)
    ap.add_argument("--exclude_docids", type=Path, default=None,
                    help="Optional JSON list of docids hidden from retrieval (no-evidence controls).")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Sampling temperature. Default model-aware: mimo 1.0, deepseek 1.0.")
    ap.add_argument("--top_p", type=float, default=None,
                    help="Nucleus top_p. Default model-aware: mimo 0.95, deepseek 1.0.")
    ap.add_argument("--snippet_tokenizer", default=_SNIPPET_TOKENIZER_PATH,
                    help="Local Qwen3 tokenizer dir for 512-token snippet truncation "
                         "(shares vocab with BCP's Qwen/Qwen3-0.6B).")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="run only first N queries (calibration)")
    ap.add_argument("--shard", default=None,
                    help="Process only a slice of the not-done queries: 'i/n' (e.g. 0/2, 1/2), "
                         "for splitting work across key-parallel processes (disjoint shards, "
                         "shared output_dir; resume-safe).")
    args = ap.parse_args()
    if args.evidence_gate in {"generated", "sourcecut"} and args.gate_obligations is not None:
        ap.error(f"{args.evidence_gate} mode forbids --gate_obligations")
    if args.gate_corruption_budget < 0:
        ap.error("--gate_corruption_budget must be non-negative")

    from openai import OpenAI

    # Pick provider endpoint by model prefix (OpenAI-compatible APIs).
    extra_headers = {}
    ml = args.model.lower()
    if ml.startswith("mimo"):
        # Two MiMo endpoints: subscription (tp- keys) at token-plan-cn, pay-as-you-go
        # (sk- keys) at api.xiaomimimo.com. Default to subscription; MIMO_BASE_URL env
        # overrides (set it to https://api.xiaomimimo.com/v1 when using an sk- key).
        base_url = os.environ.get("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
        key_env = "MIMO_API_KEY"
    elif ml.startswith(("ds", "hkgai")):
        # HKGAI gateway as the AGENT (e.g. --model DS3.2 -> deepseek-v3.2 backend).
        # OpenAI-compatible; a WAF Cookie (HKGAI_COOKIE) is sent as a default header.
        base_url = os.environ.get("HKGAI_BASE_URL", "https://api.hkgai.net/v1")
        key_env = "HKGAI_API_KEY"
        cookie = os.environ.get("HKGAI_COOKIE", "")
        if cookie:
            extra_headers["Cookie"] = cookie
    elif ml.startswith("gemini"):
        # Gemini via the xcode.best relay (OpenAI-compatible), e.g. --model gemini-2.5-flash.
        base_url = os.environ.get("XCODE_BASE_URL", "https://xcode.best/v1")
        key_env = "XCODE_API_KEY"
    elif ml.startswith("gpt-5-mini"):
        # gpt-5-mini via the vip.yinli.one relay (separate key).
        base_url = os.environ.get("GPT5MINI_BASE_URL", "https://vip.yinli.one/v1")
        key_env = "GPT5MINI_API_KEY"
    elif ml.startswith("gpt"):
        # GPT via the xcode.best relay (OpenAI-compatible), e.g. --model gpt-5.4.
        base_url = os.environ.get("GPT_BASE_URL", "https://xcode.best/v1")
        key_env = "GPT_API_KEY"
    elif ml.startswith("qwen3"):
        # Local Qwen3-32B served by vLLM (OpenAI-compatible). No key needed; free, no rate limit.
        base_url = os.environ.get("LOCAL_QWEN_URL", "http://127.0.0.1:18800/v1")
        os.environ.setdefault("LOCAL_QWEN_KEY", "EMPTY")
        key_env = "LOCAL_QWEN_KEY"
    else:  # deepseek-* (default official; DEEPSEEK_BASE_URL overrides for a relay endpoint)
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        key_env = "DEEPSEEK_API_KEY"
    api_key = os.environ.get(key_env)
    if not api_key:
        raise SystemExit(f"{key_env} not set")
    # max_retries=6 so 429s back off & ride bursts on rate-limited subscription tiers.
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=6,
                    default_headers=extra_headers or None)

    # Model-aware sampling: MiMo temp=1.0/top_p=0.95; DeepSeek temp=1.0/top_p=1.0;
    # Qwen3 (thinking) temp=0.6/top_p=0.95 (Qwen3 recommended). --temperature/--top_p override.
    _ml = args.model.lower()
    is_mimo = _ml.startswith("mimo")
    is_qwen3 = _ml.startswith("qwen3")
    temperature = args.temperature if args.temperature is not None else (0.6 if is_qwen3 else 1.0)
    top_p = args.top_p if args.top_p is not None else (0.95 if (is_mimo or is_qwen3) else 1.0)

    if args.dense_index is not None:
        if not args.bm25_index:
            raise SystemExit("--bm25_index is also required (full-text source for "
                             "search snippets when --dense_index is set)")
        searcher = DenseSearcher(
            dense_index_dir=str(args.dense_index),
            encode_server_url=args.encode_server,
            docs_lucene_index=str(args.bm25_index),
        )
        retriever_label = f"dense:{args.dense_index.name}"
    else:
        from pyserini.search.lucene import LuceneSearcher
        if not args.bm25_index:
            raise SystemExit("either --dense_index or --bm25_index must be set")
        searcher = LuceneSearcher(str(args.bm25_index))
        retriever_label = f"bm25:{args.bm25_index.name}"
    if args.exclude_docids:
        raw_excluded = json.loads(args.exclude_docids.read_text())
        excluded = raw_excluded.get("docids", []) if isinstance(raw_excluded, dict) else raw_excluded
        searcher = ExcludingSearcher(searcher, excluded)
        retriever_label += f"-excluding:{len(excluded)}"
    print(f"[{args.condition}] retriever = {retriever_label} | BCP-canonical "
          f"(max_iter={args.max_iterations}, k={args.k}, snippet={args.snippet_max_tokens}tok, "
          f"max_tokens={args.max_tokens}, search-only, temp={temperature}, top_p={top_p}, "
          f"gate={args.evidence_gate})", flush=True)

    obligation_data = None
    if args.gate_obligations:
        obligation_data = json.loads(args.gate_obligations.read_text())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    queries = [json.loads(line) for line in args.queries.open()]
    if args.limit:
        queries = queries[: args.limit]

    # Resume: only skip successful runs. Error JSONs may contain diagnostic
    # output_text, but they must be retried by the outer pass loop.
    def done(qid):
        p = args.output_dir / f"{qid}.json"
        if not p.exists():
            return False
        try:
            rd = json.loads(p.read_text())
            return rd.get("status") == "completed"
        except Exception:
            return False

    todo = [q for q in queries if not done(q["query_id"])]
    n_done = len(queries) - len(todo)
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        todo = todo[si::sn]
        print(f"[{args.condition}] shard {si}/{sn} -> {len(todo)} queries this process "
              f"({n_done} already done overall)", flush=True)
    else:
        print(f"[{args.condition}] {len(todo)} to run, {n_done} already done", flush=True)

    def work(q):
        if args.model.lower().startswith("gemini"):
            if args.evidence_gate != "none":
                raise ValueError("evidence gates are not yet supported by the Gemini native runner")
            # Gemini 3.x needs native generateContent to preserve thoughtSignature
            # (OpenAI-compat relays drop it -> degraded multi-step tool use).
            r = run_one_gemini_native(
                    query_id=str(q["query_id"]), query=q["query"], condition=args.condition,
                    searcher=searcher, api_key=api_key, base_url=base_url, model=args.model,
                    max_iterations=args.max_iterations, k=args.k,
                    snippet_max_tokens=args.snippet_max_tokens,
                    snippet_tokenizer_path=args.snippet_tokenizer,
                    temperature=temperature, top_p=top_p, max_tokens=args.max_tokens,
                    instruction_variant=args.instruction_variant)
        else:
            r = run_one(query_id=str(q["query_id"]), query=q["query"], condition=args.condition,
                    searcher=searcher, client=client, model=args.model,
                    max_iterations=args.max_iterations, k=args.k,
                    snippet_max_tokens=args.snippet_max_tokens,
                    snippet_tokenizer_path=args.snippet_tokenizer,
                    temperature=temperature, top_p=top_p, max_tokens=args.max_tokens,
                    instruction_variant=args.instruction_variant,
                    evidence_gate_mode=args.evidence_gate,
                    gate_model=args.gate_model,
                    gate_obligations=(
                        obligation_data.get(str(q["query_id"]))
                        if obligation_data and "obligations" not in obligation_data
                        else obligation_data
                    ),
                    gate_max_rejections=args.gate_max_rejections,
                    gate_stagnation_limit=args.gate_stagnation_limit,
                    gate_corruption_budget=args.gate_corruption_budget)
        (args.output_dir / f"{q['query_id']}.json").write_text(json.dumps(r, ensure_ascii=False))
        return r

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, q): q for q in todo}
        for fut in as_completed(futs):
            q = futs[fut]
            try:
                r = fut.result()
                n += 1
                txt = next((e["output"] for e in reversed(r["result"]) if e["type"] == "output_text"), "")
                print(f"[{args.condition}] {n}/{len(todo)} qid={r['query_id']} "
                      f"calls={r['tool_call_counts']} status={r['status']} ans={txt[:70]!r}", flush=True)
            except Exception as e:
                print(f"[{args.condition}] qid={q['query_id']} FAILED: {type(e).__name__}: {e}", flush=True)

    print(f"[{args.condition}] done: {n} new runs -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
