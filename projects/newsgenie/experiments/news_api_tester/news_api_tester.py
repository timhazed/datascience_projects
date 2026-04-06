"""
News API Tester — Measures latency and errors for news + web search APIs using labeled JSONL.

Parses `ground_truth` in the same format as `Intent_Validation_Mini.jsonl` (INTENT: … | QUERY: …),
routes each sub-query to NewsAPI `/v2/everything` (Business + General), Guardian search (Sports), or web search (SerpAPI / Serper) — matching requirements`.

Web search uses LangChain community wrappers (same stack as LangGraph tooling): `SerpAPIWrapper`,
`GoogleSerperAPIWrapper`. Exactly **one** provider runs per run, selected by `WEB_SEARCH_PROVIDER`
in `.env` (same idea as `LLM_PROVIDER`).

Usage:
    poetry run news-api-tester --input ../../data/Intent_Validation_Mini.jsonl
    poetry run python experiments/news_api_tester/news_api_tester.py --input data/foo.jsonl

Loads API keys from `.env` (project root): GUARDIAN_API_KEY, NEWSAPI_API_KEY,
SERPAPI_API_KEY, SERPER_API_KEY (or SERPERDEV_API_KEY). See `.env.example`.

Statistics JSON: `experiments/news_api_tester/data/News_API_Stats_<WEB_SEARCH_PROVIDER>_<YYYYMMDD_HHMMSS>.json`
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper, SerpAPIWrapper

# ---------------------------------------------------------------------------
# Ground-truth parsing (same convention as intent_tester)
# ---------------------------------------------------------------------------

VALID_INTENTS = {"Business News", "Sports News", "General News", "Web Search"}
INTENT_PATTERN = re.compile(r"INTENT:\s*(.*?)\s*\|\s*QUERY:\s*(.*)", re.MULTILINE)

INTENT_TO_API = {
    "Business News": "newsapi",
    "Sports News": "guardian",
    "General News": "newsapi",
    "Web Search": "web_search",
}

# ---------------------------------------------------------------------------
# HTTP clients — minimal GETs matching ARCHITECTURE.md providers
# ---------------------------------------------------------------------------

TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def _clip_q(q: str, max_len: int = 500) -> str:
    q = q.strip()
    return q[:max_len] if len(q) > max_len else q


def call_guardian(client: httpx.Client, api_key: str, query: str) -> tuple[int, str | None]:
    url = "https://content.guardianapis.com/search"
    params = {
        "api-key": api_key,
        "q": _clip_q(query),
        "page-size": "5",
        "show-fields": "headline",
    }
    r = client.get(url, params=params, timeout=TIMEOUT)
    err = None if r.is_success else (r.text[:500] if r.text else r.reason_phrase)
    return r.status_code, err


def call_newsapi(client: httpx.Client, api_key: str, query: str) -> tuple[int, str | None]:
    """GET NewsAPI `/v2/everything` — used for Business News and General News per requirements."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": api_key,
        "q": _clip_q(query),
        "pageSize": 5,
        "language": "en",
        "sortBy": "publishedAt",
    }
    r = client.get(url, params=params, timeout=TIMEOUT)
    err = None if r.is_success else (r.text[:500] if r.text else r.reason_phrase)
    return r.status_code, err


def parse_web_search_provider() -> str:
    """Single active web search backend, analogous to LLM_PROVIDER."""
    raw = os.getenv("WEB_SEARCH_PROVIDER", "serpapi").strip().lower()
    if raw in ("serpapi", "serper"):
        return raw
    return "serpapi"


def _serpapi_processed_ok(text: Any) -> bool:
    """
    LangChain SerpAPIWrapper._process_response returns str, list, or dict — not only str.
    Treat any non-empty structured result as success.
    """
    if text is None:
        return False
    if isinstance(text, str):
        return bool(text.strip())
    if isinstance(text, (list | tuple | dict | set)):
        return len(text) > 0
    return True


def _serpapi_failure_detail(text: Any, raw: dict[str, Any]) -> str:
    keys = list(raw.keys())[:35] if isinstance(raw, dict) else []
    bits = [
        f"processed_type={type(text).__name__}",
        f"processed_repr={repr(text)[:600]}",
        f"raw_top_level_keys={keys}",
    ]
    if isinstance(raw, dict) and raw.get("error"):
        bits.append(f"serpapi_error={raw['error']!r}")
    return " | ".join(bits)


def _serper_api_message_if_error(raw: dict[str, Any]) -> str | None:
    """Serper.dev sometimes returns error text in JSON (e.g. invalid key) alongside HTTP 200."""
    if not isinstance(raw, dict):
        return None
    for key in ("message", "error", "errors"):
        val = raw.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, list) and val and isinstance(val[0], str):
            return val[0].strip()
    return None


def _serper_failure_detail(text: str, raw: dict[str, Any]) -> str:
    keys = list(raw.keys())[:35] if isinstance(raw, dict) else []
    bits = [
        f"parsed_text_len={len(text)}",
        f"parsed_text_repr={repr(text)[:600]}",
        f"raw_top_level_keys={keys}",
    ]
    m = _serper_api_message_if_error(raw)
    if m:
        bits.append(f"serper_message_field={m!r}")
    return " | ".join(bits)


def run_web_search_langchain(
    query: str,
    keys: dict[str, str],
    provider: str,
) -> dict[str, Any]:
    """
    Run exactly one of SerpAPI or Serper per `WEB_SEARCH_PROVIDER`.
    Result always uses api=web_search and web_search_provider for summary bucketing.
    """
    q = _clip_q(query)
    ws = {"api": "web_search", "web_search_provider": provider}

    if provider == "serpapi":
        k = keys.get("serpapi") or ""
        if not k:
            return {
                **ws,
                "success": False,
                "http_status": None,
                "latency_ms": None,
                "error": None,
                "skip_reason": "SERPAPI_API_KEY not set (WEB_SEARCH_PROVIDER=serpapi)",
            }
        t0 = time.perf_counter()
        raw: dict[str, Any] | None = None
        try:
            wrapper = SerpAPIWrapper(serpapi_api_key=k)
            raw = wrapper.results(q)
            text = SerpAPIWrapper._process_response(raw)
        except ValueError as e:
            # SerpAPI JSON includes an "error" key; _process_response raises with message.
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                **ws,
                "success": False,
                "http_status": None,
                "latency_ms": round(latency_ms, 2),
                "error": str(e),
                "skip_reason": None,
                "serpapi_response_keys": list(raw.keys()) if isinstance(raw, dict) else None,
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                **ws,
                "success": False,
                "http_status": None,
                "latency_ms": round(latency_ms, 2),
                "error": repr(e),
                "skip_reason": None,
                "serpapi_response_keys": list(raw.keys()) if isinstance(raw, dict) else None,
            }

        latency_ms = (time.perf_counter() - t0) * 1000
        ok = _serpapi_processed_ok(text)
        if not ok:
            detail = _serpapi_failure_detail(text, raw or {})
            return {
                **ws,
                "success": False,
                "http_status": None,
                "latency_ms": round(latency_ms, 2),
                "error": f"SerpAPI returned no usable result ({detail})",
                "skip_reason": None,
                "serpapi_response_keys": list(raw.keys())[:40] if isinstance(raw, dict) else None,
            }
        return {
            **ws,
            "success": True,
            "http_status": 200,
            "latency_ms": round(latency_ms, 2),
            "error": None,
            "skip_reason": None,
        }

    elif provider == "serper":
        k = keys.get("serper") or ""
        if not k:
            return {
                **ws,
                "success": False,
                "http_status": None,
                "latency_ms": None,
                "error": None,
                "skip_reason": "SERPER_API_KEY or SERPERDEV_API_KEY not set (WEB_SEARCH_PROVIDER=serper)",
            }
        t0 = time.perf_counter()
        raw: dict[str, Any] | None = None
        try:
            wrapper = GoogleSerperAPIWrapper(serper_api_key=k)
            raw = wrapper.results(q)
            text = wrapper._parse_results(raw)
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                **ws,
                "success": False,
                "http_status": None,
                "latency_ms": round(latency_ms, 2),
                "error": repr(e),
                "skip_reason": None,
                "serper_response_keys": list(raw.keys()) if isinstance(raw, dict) else None,
            }

        latency_ms = (time.perf_counter() - t0) * 1000
        api_msg = _serper_api_message_if_error(raw or {})
        if api_msg:
            return {
                **ws,
                "success": False,
                "http_status": None,
                "latency_ms": round(latency_ms, 2),
                "error": f"Serper API: {api_msg}",
                "skip_reason": None,
                "serper_response_keys": list(raw.keys())[:40] if isinstance(raw, dict) else None,
            }
        if not (isinstance(text, str) and text.strip()):
            detail = _serper_failure_detail(text if isinstance(text, str) else "", raw or {})
            return {
                **ws,
                "success": False,
                "http_status": None,
                "latency_ms": round(latency_ms, 2),
                "error": f"Serper returned no usable text ({detail})",
                "skip_reason": None,
                "serper_response_keys": list(raw.keys())[:40] if isinstance(raw, dict) else None,
            }
        return {
            **ws,
            "success": True,
            "http_status": 200,
            "latency_ms": round(latency_ms, 2),
            "error": None,
            "skip_reason": None,
        }

    return {
        **ws,
        "success": False,
        "http_status": None,
        "latency_ms": None,
        "error": f"unknown WEB_SEARCH_PROVIDER {provider!r}",
        "skip_reason": None,
    }


def parse_ground_truth(text: str) -> list[dict[str, str]]:
    matches = INTENT_PATTERN.findall(text)
    out: list[dict[str, str]] = []
    for intent, query in matches:
        label = intent.strip()
        q = query.strip()
        if label not in VALID_INTENTS:
            label = "Web Search"
        out.append({"intent": label, "query": q})
    return out


def percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    data = sorted(sorted_vals)
    k = (len(data) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(data[int(k)])
    return float(data[f] + (k - f) * (data[c] - data[f]))


def run_one_call_with_latency(
    client: httpx.Client,
    api: str,
    keys: dict[str, str],
    query: str,
    web_search_provider: str,
) -> dict[str, Any]:
    """Single API call with latency_ms always set when HTTP attempted."""
    if api == "web_search":
        return run_web_search_langchain(query, keys, web_search_provider)

    if api == "guardian":
        k = keys.get("guardian") or ""
        if not k:
            return {
                "success": False,
                "http_status": None,
                "latency_ms": None,
                "error": None,
                "skip_reason": "GUARDIAN_API_KEY not set",
            }
        t0 = time.perf_counter()
        try:
            status, err = call_guardian(client, k, query)
            latency_ms = (time.perf_counter() - t0) * 1000
            ok = 200 <= status < 300
            return {
                "success": ok,
                "http_status": status,
                "latency_ms": round(latency_ms, 2),
                "error": err if not ok else None,
                "skip_reason": None,
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "success": False,
                "http_status": None,
                "latency_ms": round(latency_ms, 2),
                "error": repr(e),
                "skip_reason": None,
            }

    if api == "newsapi":
        k = keys.get("newsapi") or ""
        if not k:
            return {
                "success": False,
                "http_status": None,
                "latency_ms": None,
                "error": None,
                "skip_reason": "NEWSAPI_API_KEY not set",
            }
        t0 = time.perf_counter()
        try:
            status, err = call_newsapi(client, k, query)
            latency_ms = (time.perf_counter() - t0) * 1000
            ok = 200 <= status < 300
            return {
                "success": ok,
                "http_status": status,
                "latency_ms": round(latency_ms, 2),
                "error": err if not ok else None,
                "skip_reason": None,
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "success": False,
                "http_status": None,
                "latency_ms": round(latency_ms, 2),
                "error": repr(e),
                "skip_reason": None,
            }

    return {
        "success": False,
        "http_status": None,
        "latency_ms": None,
        "error": f"unknown api {api}",
        "skip_reason": None,
    }


PROMPT_LOG_MAX = 4000


def _emit_call_issue(
    call_id: int,
    prompt: str,
    api: str | None,
    query: str | None,
    result: dict[str, Any],
) -> None:
    """Print source prompt and error/skip details to stderr (failures and skips)."""
    if result.get("success") and not result.get("skip_reason"):
        return
    parts = [
        f"--- news-api-tester issue (call_id={call_id}) ---",
        f"api={api!r}",
    ]
    if query is not None:
        parts.append(f"query={query!r}")
    if api == "web_search" or result.get("web_search_provider"):
        parts.append(f"web_search_provider={result.get('web_search_provider')!r}")
    err = result.get("error")
    sk = result.get("skip_reason")
    if err:
        parts.append(f"error={err}")
    if sk:
        parts.append(f"skip_reason={sk}")
    if result.get("http_status") is not None:
        parts.append(f"http_status={result['http_status']}")
    if result.get("serpapi_response_keys"):
        parts.append(f"serpapi_response_keys={result['serpapi_response_keys']!r}")
    if result.get("serper_response_keys"):
        parts.append(f"serper_response_keys={result['serper_response_keys']!r}")
    p = prompt if len(prompt) <= PROMPT_LOG_MAX else prompt[: PROMPT_LOG_MAX - 3] + "..."
    parts.append(f"prompt:\n{p}")
    print("\n".join(parts), file=sys.stderr, flush=True)


def summarize_by_api(
    calls: list[dict[str, Any]],
    web_search_provider: str,
) -> dict[str, Any]:
    by_api: dict[str, list[dict[str, Any]]] = {
        "guardian": [],
        "newsapi": [],
        "web_search": [],
    }
    for c in calls:
        api = c.get("api")
        if api in by_api:
            by_api[api].append(c)

    summary: dict[str, Any] = {}
    for api, rows in by_api.items():
        latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        successes = sum(1 for r in rows if r.get("success"))
        skips = sum(1 for r in rows if r.get("skip_reason"))
        failures = len(rows) - successes - skips
        block: dict[str, Any] = {
            "total_calls": len(rows),
            "successes": successes,
            "failures_http_or_error": failures,
            "skipped_missing_key": skips,
            "latency_ms": {
                "count": len(latencies),
                "min": min(latencies) if latencies else None,
                "max": max(latencies) if latencies else None,
                "mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
                "p50": round(percentile(latencies, 50), 2) if latencies else None,
                "p95": round(percentile(latencies, 95), 2) if latencies else None,
                "p99": round(percentile(latencies, 99), 2) if latencies else None,
            },
        }
        if api == "web_search":
            block["provider"] = web_search_provider
        summary[api] = block
    return summary


def latency_long_poles(summary_by_api: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Rank APIs/tools by mean latency (slowest first) to highlight long poles.
    Entries with no measured latency are omitted.
    """
    rows: list[dict[str, Any]] = []
    for name, block in summary_by_api.items():
        mean = block.get("latency_ms", {}).get("mean")
        if mean is None:
            continue
        rows.append(
            {
                "api": name,
                "mean_latency_ms": mean,
                "p95_latency_ms": block["latency_ms"].get("p95"),
                "total_calls": block.get("total_calls", 0),
            }
        )
    rows.sort(key=lambda x: x["mean_latency_ms"], reverse=True)
    return rows


def find_dotenv() -> Path | None:
    here = Path(__file__).resolve().parent
    for base in [here, *here.parents]:
        candidate = base / ".env"
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark news APIs + web search (LangChain SerpAPI / Serper) from JSONL."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSONL (prompt + ground_truth fields per line)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for Statistics JSON (default: experiments/news_api_tester/data)",
    )
    args = parser.parse_args()

    env_path = find_dotenv()
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else script_dir / "data"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = {
        "guardian": os.getenv("GUARDIAN_API_KEY", "").strip(),
        "newsapi": os.getenv("NEWSAPI_API_KEY", "").strip(),
        "serpapi": os.getenv("SERPAPI_API_KEY", "").strip(),
        "serper": (
            os.getenv("SERPER_API_KEY", "").strip()
            or os.getenv("SERPERDEV_API_KEY", "").strip()
        ),
    }
    web_search_provider = parse_web_search_provider()

    records: list[dict[str, Any]] = []
    with input_path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: line {line_no} invalid JSON: {e}", file=sys.stderr)

    jobs: list[tuple[int, str, dict[str, str] | None]] = []
    for rec_idx, rec in enumerate(records):
        prompt = rec.get("prompt", "")
        gt = rec.get("ground_truth", "")
        steps = parse_ground_truth(gt)
        if not steps:
            jobs.append((rec_idx, prompt, None))
        else:
            for step in steps:
                jobs.append((rec_idx, prompt, step))

    total_jobs = len(jobs)
    calls: list[dict[str, Any]] = []
    call_id = 0

    t_run = time.perf_counter()
    with httpx.Client(headers={"User-Agent": "NewsGenie-news-api-tester/1.0"}) as client:
        for jidx, (rec_idx, prompt, step) in enumerate(jobs, 1):
            if step is None:
                call_id += 1
                result = {
                    "success": False,
                    "http_status": None,
                    "latency_ms": None,
                    "error": "no INTENT|QUERY pairs in ground_truth",
                    "skip_reason": None,
                }
                _emit_call_issue(call_id, prompt, None, None, result)
                calls.append(
                    {
                        "call_id": call_id,
                        "record_index": rec_idx,
                        "prompt": prompt,
                        "intent": None,
                        "api": None,
                        "query": None,
                        **result,
                    }
                )
                print(
                    f"[{jidx}/{total_jobs}] record={rec_idx} skipped (no INTENT|QUERY pairs)",
                    flush=True,
                )
                continue

            intent = step["intent"]
            query = step["query"]
            api = INTENT_TO_API[intent]
            call_id += 1
            print(
                f"[{jidx}/{total_jobs}] api={api} record={rec_idx} query={query[:80]!r}"
                + (" …" if len(query) > 80 else ""),
                flush=True,
            )
            result = run_one_call_with_latency(
                client,
                api,
                keys,
                query,
                web_search_provider,
            )
            _emit_call_issue(call_id, prompt, api, query, result)
            row = {
                "call_id": call_id,
                "record_index": rec_idx,
                "prompt": prompt,
                "intent": intent,
                "api": api,
                "query": query[:300],
                **result,
            }
            calls.append(row)

    elapsed_total_s = round(time.perf_counter() - t_run, 2)
    summary = summarize_by_api(calls, web_search_provider)
    long_poles = latency_long_poles(summary)

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_name = f"News_API_Stats_{web_search_provider}_{ts}.json"
    out_path = out_dir / out_name

    payload = {
        "run_timestamp": datetime.now(UTC).isoformat(),
        "input_file": str(input_path),
        "output_file": str(out_path),
        "elapsed_total_seconds": elapsed_total_s,
        "dotenv_loaded_from": str(env_path) if env_path else None,
        "keys_configured": {k: bool(v) for k, v in keys.items()},
        "web_search_provider": web_search_provider,
        "summary_by_api": summary,
        "latency_long_poles_mean_ms_desc": long_poles,
        "calls": calls,
    }

    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"WEB_SEARCH_PROVIDER={web_search_provider}")
    print(f"Wrote statistics → {out_path}")
    print(json.dumps(summary, indent=2))
    print("\nLatency long poles (mean ms, slowest first):", flush=True)
    for row in long_poles:
        print(
            f"  {row['api']}: mean={row['mean_latency_ms']} ms  "
            f"p95={row['p95_latency_ms']}  n={row['total_calls']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
