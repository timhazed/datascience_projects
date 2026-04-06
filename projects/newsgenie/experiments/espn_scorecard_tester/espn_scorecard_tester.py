"""
ESPN scoreboard tester — JSONL-driven experiment for `site.api.espn.com` scoreboards.

Fetches US-oriented league scoreboards (NFL, NBA, MLB, NHL), parses JSON, and **vets**
each response against optional `expect` rules in the JSONL (see `data/scoreboard_cases.jsonl`).

**Why:** ESPN as a secondary sports source
for scores/standings where Guardian coverage is weak for North American leagues. This
harness measures latency, shape of payloads, and whether returned content matches
declared expectations (no API key; **undocumented** endpoints — be polite: no rapid polling).

Base URL pattern:
  https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard

Query params (examples): `dates=YYYYMMDD`, `dates=YYYYMMDD-YYYYMMDD`, `limit=N`

Usage:
  poetry run espn-scorecard-tester
  poetry run espn-scorecard-tester --input path/to/cases.jsonl -v
  poetry run python -m experiments.espn_scorecard_tester

Output JSON — **one pair per JSONL line**, sharing the same run timestamp:
  - `ESPN_Scoreboard_Stats_<league>_<case_id>_<YYYYMMDD_HHMMSS>.json` — vetting, previews, latency for that query
  - `ESPN_Scoreboard_Results_<league>_<case_id>_<YYYYMMDD_HHMMSS>.json` — full ESPN scoreboard body for that query

`<league>` and `<case_id>` are sanitized from the JSONL row (two NBA rows → two distinct files).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

TIMEOUT = httpx.Timeout(30.0, connect=10.0)
USER_AGENT = "NewsGenie-espn-scorecard-tester/1.0 (research; low frequency)"

# sport slug, league slug — maps JSONL `league` key (lowercase)
LEAGUE_TO_PATH: dict[str, tuple[str, str]] = {
    "nfl": ("football", "nfl"),
    "nba": ("basketball", "nba"),
    "mlb": ("baseball", "mlb"),
    "nhl": ("hockey", "nhl"),
}

BASE = "https://site.api.espn.com/apis/site/v2/sports"


def safe_filename_part(s: str, max_len: int = 100) -> str:
    """Sanitize JSONL `league` / `id` segments for use in filenames."""
    t = re.sub(r"[^a-zA-Z0-9._-]+", "_", (s or "").strip())
    t = re.sub(r"_+", "_", t).strip("_") or "case"
    return t[:max_len]


def summary_for_single_call(call: dict[str, Any]) -> dict[str, Any]:
    """Aggregate block for a stats file that describes exactly one query."""
    lat = call.get("latency_ms")
    if lat is None:
        latency_block: dict[str, Any] = {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p95": None,
        }
    else:
        latency_block = {
            "count": 1,
            "min": lat,
            "max": lat,
            "mean": lat,
            "p50": lat,
            "p95": lat,
        }
    ok = bool(call.get("success"))
    vok = bool(call.get("vet_passed"))
    return {
        "cases_total": 1,
        "http_success": 1 if ok else 0,
        "http_failed": 0 if ok else 1,
        "vet_passed": 1 if vok else 0,
        "vet_failed": 0 if vok else 1,
        "latency_ms": latency_block,
    }


def write_case_outputs(
    out_dir: Path,
    ts_suffix: str,
    run_ts_iso: str,
    input_path: Path,
    call: dict[str, Any],
    result_row: dict[str, Any],
) -> tuple[Path, Path]:
    """Write Results + Stats JSON for one case; return (stats_path, results_path)."""
    league = str(call.get("league") or "unknown")
    case_id = str(call.get("case_id") or "case")
    ls = safe_filename_part(league)
    cs = safe_filename_part(case_id)
    stats_name = f"ESPN_Scoreboard_Stats_{ls}_{cs}_{ts_suffix}.json"
    results_name = f"ESPN_Scoreboard_Results_{ls}_{cs}_{ts_suffix}.json"
    stats_path = out_dir / stats_name
    results_path = out_dir / results_name

    results_payload = {
        "run_timestamp": run_ts_iso,
        "input_file": str(input_path),
        "stats_file": str(stats_path),
        **result_row,
    }
    stats_payload = {
        "run_timestamp": run_ts_iso,
        "input_file": str(input_path),
        "output_file": str(stats_path),
        "scoreboard_results_file": str(results_path),
        "case_id": call.get("case_id"),
        "league": call.get("league"),
        "summary": summary_for_single_call(call),
        "calls": [call],
        "notes": (
            "Undocumented ESPN APIs — use low request frequency. "
            "Vet rules are per-line `expect` in JSONL; see data/scoreboard_cases.jsonl. "
            "Full response body: `scoreboard_results_file`."
        ),
    }

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)
    return stats_path, results_path


def scoreboard_url(league_key: str) -> str:
    sport, lg = LEAGUE_TO_PATH[league_key.lower()]
    return f"{BASE}/{sport}/{lg}/scoreboard"


def scoreboard_request_url(
    league_key: str, dates: str | None, limit: int | None
) -> str:
    """GET URL including query string (for logs and results JSON)."""
    url = scoreboard_url(league_key)
    params: dict[str, str] = {}
    if dates:
        params["dates"] = dates
    if limit is not None:
        params["limit"] = str(limit)
    if not params:
        return url
    return f"{url}?{urlencode(params)}"


def collect_status_names(data: dict[str, Any]) -> set[str]:
    """All competition status.type.name values across events."""
    names: set[str] = set()
    for ev in data.get("events") or []:
        for comp in ev.get("competitions") or []:
            nm = (comp.get("status") or {}).get("type", {}).get("name")
            if isinstance(nm, str) and nm:
                names.add(nm)
    return names


def summarize_events(data: dict[str, Any], max_preview: int = 5) -> list[dict[str, Any]]:
    """Lightweight per-event summary for the stats JSON."""
    out: list[dict[str, Any]] = []
    for ev in (data.get("events") or [])[:max_preview]:
        row: dict[str, Any] = {
            "id": ev.get("id"),
            "shortName": ev.get("shortName"),
            "name": ev.get("name"),
        }
        comps = ev.get("competitions") or []
        if comps:
            c0 = comps[0]
            row["status"] = (c0.get("status") or {}).get("type", {}).get("name")
            teams = []
            for side in c0.get("competitors") or []:
                tm = side.get("team") or {}
                teams.append(
                    {
                        "name": tm.get("displayName") or tm.get("name"),
                        "abbrev": tm.get("abbreviation"),
                        "score": side.get("score"),
                    }
                )
            row["teams"] = teams
        out.append(row)
    return out


def vet_case(parsed: dict[str, Any], expect: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Return (passed, failure_reasons). `expect` is optional; empty/missing => pass.
    """
    reasons: list[str] = []
    if not expect:
        return True, []

    for key in expect.get("keys_present") or []:
        if key not in parsed:
            reasons.append(f"missing top-level key {key!r}")

    events = parsed.get("events")
    if events is None:
        reasons.append("missing 'events' array")
        n = 0
    else:
        n = len(events)

    min_e = expect.get("min_events")
    if min_e is not None and n < min_e:
        reasons.append(f"events count {n} < min_events {min_e}")

    max_e = expect.get("max_events")
    if max_e is not None and n > max_e:
        reasons.append(f"events count {n} > max_events {max_e}")

    allowed = expect.get("status_types_any")
    if allowed and n > 0:
        found = collect_status_names(parsed)
        if not found.intersection(set(allowed)):
            reasons.append(
                f"no competition status in allowed set {allowed!r}; found {sorted(found)}"
            )

    for sub in expect.get("response_text_contains") or []:
        blob = json.dumps(parsed).lower()
        if sub.lower() not in blob:
            reasons.append(f"response JSON does not contain substring {sub!r}")

    for sub in expect.get("team_substrings_any") or []:
        if n == 0:
            continue
        blob = json.dumps(parsed).lower()
        if sub.lower() not in blob:
            reasons.append(f"no team/text match for substring {sub!r}")

    return len(reasons) == 0, reasons


def fetch_scoreboard(
    client: httpx.Client,
    league_key: str,
    dates: str | None,
    limit: int | None,
) -> tuple[int, dict[str, Any] | None, str | None]:
    """GET scoreboard; return (status_code, json_dict_or_none, error_message)."""
    url = scoreboard_url(league_key)
    params: dict[str, Any] = {}
    if dates:
        params["dates"] = dates
    if limit is not None:
        params["limit"] = limit
    try:
        r = client.get(url, params=params or None, timeout=TIMEOUT)
        if not r.is_success:
            return r.status_code, None, (r.text[:800] if r.text else r.reason_phrase)
        data = r.json()
        if not isinstance(data, dict):
            return r.status_code, None, f"JSON root is {type(data).__name__}, expected object"
        return r.status_code, data, None
    except Exception as e:
        return 0, None, repr(e)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ESPN site.api scoreboard harness — JSONL cases + vetting."
    )
    script_dir = Path(__file__).resolve().parent
    default_jsonl = script_dir / "data" / "scoreboard_cases.jsonl"
    parser.add_argument(
        "--input",
        default=str(default_jsonl),
        help=f"Path to JSONL (one test case per line); default: {default_jsonl}",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output stats JSON (default: experiments/espn_scorecard_tester/data)",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.75,
        help="Sleep between requests to reduce risk of rate limiting (default: 0.75)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print one line per case (id, HTTP, vet, event count)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else script_dir / "data"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: line {line_no} invalid JSON: {e}", file=sys.stderr)

    if not records:
        print("ERROR: No JSONL records loaded.", file=sys.stderr)
        sys.exit(1)

    calls: list[dict[str, Any]] = []
    t_run = time.perf_counter()
    ts_suffix = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_ts_iso = datetime.now(UTC).isoformat()

    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        for idx, rec in enumerate(records):
            case_id = rec.get("id", f"line_{idx}")
            league = (rec.get("league") or "").strip().lower()
            if league not in LEAGUE_TO_PATH:
                err_msg = (
                    f"unknown league {league!r}; expected one of {list(LEAGUE_TO_PATH)}"
                )
                call_dict: dict[str, Any] = {
                    "case_id": case_id,
                    "league": league,
                    "success": False,
                    "vet_passed": False,
                    "latency_ms": None,
                    "http_status": None,
                    "error": err_msg,
                    "vet_reasons": [],
                    "event_count": None,
                    "status_names": [],
                    "events_preview": [],
                }
                result_dict: dict[str, Any] = {
                    "case_id": case_id,
                    "league": league,
                    "description": rec.get("description", ""),
                    "request_url": None,
                    "dates": None,
                    "limit": rec.get("limit"),
                    "success": False,
                    "http_status": None,
                    "latency_ms": None,
                    "error": err_msg,
                    "vet_passed": False,
                    "vet_reasons": [],
                    "scoreboard": None,
                }
                calls.append(call_dict)
                stats_path, results_path = write_case_outputs(
                    out_dir, ts_suffix, run_ts_iso, input_path, call_dict, result_dict
                )
                print(f"Wrote → {results_path}")
                print(f"Wrote → {stats_path}")
                if args.verbose:
                    c = call_dict
                    print(
                        f"{c['case_id']}: league={c.get('league')} http={c.get('http_status')} "
                        f"events={c.get('event_count')} vet={'ok' if c.get('vet_passed') else 'FAIL'} "
                        f"{c.get('latency_ms')}ms err={c.get('error')}",
                        file=sys.stderr,
                    )
                if idx < len(records) - 1 and args.delay_seconds > 0:
                    time.sleep(args.delay_seconds)
                continue

            dates = rec.get("dates")
            if dates is not None and dates != "":
                dates = str(dates).strip()
            else:
                dates = None

            limit = rec.get("limit")
            if limit is not None:
                limit = int(limit)

            expect = rec.get("expect") or {}
            desc = rec.get("description", "")

            req_url = scoreboard_request_url(league, dates, limit)
            t0 = time.perf_counter()
            status, parsed, err = fetch_scoreboard(client, league, dates, limit)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            if err or parsed is None:
                call_dict = {
                    "case_id": case_id,
                    "league": league,
                    "description": desc,
                    "dates": dates,
                    "limit": limit,
                    "request_url": req_url,
                    "success": False,
                    "vet_passed": False,
                    "latency_ms": latency_ms,
                    "http_status": status,
                    "error": err,
                    "vet_reasons": [],
                    "event_count": None,
                    "status_names": [],
                    "events_preview": [],
                }
                result_dict = {
                    "case_id": case_id,
                    "league": league,
                    "description": desc,
                    "request_url": req_url,
                    "dates": dates,
                    "limit": limit,
                    "success": False,
                    "http_status": status,
                    "latency_ms": latency_ms,
                    "error": err,
                    "vet_passed": False,
                    "vet_reasons": [],
                    "scoreboard": None,
                }
            else:
                ok, vet_reasons = vet_case(parsed, expect)
                statuses = sorted(collect_status_names(parsed))
                n_ev = len(parsed.get("events") or [])
                call_dict = {
                    "case_id": case_id,
                    "league": league,
                    "description": desc,
                    "dates": dates,
                    "limit": limit,
                    "request_url": req_url,
                    "success": True,
                    "vet_passed": ok,
                    "latency_ms": latency_ms,
                    "http_status": status,
                    "error": None,
                    "vet_reasons": vet_reasons,
                    "event_count": n_ev,
                    "status_names": statuses,
                    "events_preview": summarize_events(parsed),
                }
                result_dict = {
                    "case_id": case_id,
                    "league": league,
                    "description": desc,
                    "request_url": req_url,
                    "dates": dates,
                    "limit": limit,
                    "success": True,
                    "http_status": status,
                    "latency_ms": latency_ms,
                    "error": None,
                    "vet_passed": ok,
                    "vet_reasons": vet_reasons,
                    "scoreboard": parsed,
                }

            calls.append(call_dict)
            stats_path, results_path = write_case_outputs(
                out_dir, ts_suffix, run_ts_iso, input_path, call_dict, result_dict
            )
            print(f"Wrote → {results_path}")
            print(f"Wrote → {stats_path}")

            if args.verbose:
                c = call_dict
                err = c.get("error")
                err_short = err
                if isinstance(err, str) and len(err) > 100:
                    err_short = err[:100] + "…"
                base = (
                    f"{c['case_id']}: league={c.get('league')} http={c.get('http_status')} "
                    f"events={c.get('event_count')} vet={'ok' if c.get('vet_passed') else 'FAIL'} "
                    f"{c.get('latency_ms')}ms"
                )
                if err_short:
                    base += f" err={err_short}"
                print(base, file=sys.stderr)
                if not c.get("vet_passed") and c.get("vet_reasons"):
                    for vr in c["vet_reasons"]:
                        print(f"  vet: {vr}", file=sys.stderr)

            if idx < len(records) - 1 and args.delay_seconds > 0:
                time.sleep(args.delay_seconds)

    elapsed = round(time.perf_counter() - t_run, 2)
    vet_ok = sum(1 for c in calls if c.get("vet_passed"))
    http_ok = sum(1 for c in calls if c.get("success"))
    latencies = [c["latency_ms"] for c in calls if c.get("latency_ms") is not None]

    summary = {
        "cases_total": len(calls),
        "http_success": http_ok,
        "http_failed": len(calls) - http_ok,
        "vet_passed": vet_ok,
        "vet_failed": len(calls) - vet_ok,
        "elapsed_total_seconds": elapsed,
        "latency_ms": {
            "count": len(latencies),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "p50": round(percentile(latencies, 50), 2) if latencies else None,
            "p95": round(percentile(latencies, 95), 2) if latencies else None,
        },
    }

    print(f"Run filename suffix (shared by this run): {ts_suffix}")
    print(json.dumps(summary, indent=2))
    if summary["vet_failed"] or summary["http_failed"]:
        print(
            f"\nWARNING: vet_failed={summary['vet_failed']} http_failed={summary['http_failed']}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
