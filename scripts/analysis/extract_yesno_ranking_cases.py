#!/usr/bin/env python3
"""Extract qualitative success and failure cases from Yes/No scorer ranking."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def truncate(text: Any, limit: int) -> str:
    value = str(text or "").replace("\n", " ")
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def fmt_float(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6f}"


def format_history(history: list[dict[str, Any]], limit: int) -> str:
    if not history:
        return "None"
    chunks = []
    for item in history[-2:]:
        action = str(item.get("action", ""))
        obs = truncate(item.get("observation", ""), limit)
        chunks.append(f"- action: `{action}`\n  observation: {obs}")
    return "\n".join(chunks)


def format_case(
    case: dict[str, Any],
    state: dict[str, Any] | None,
    obs_chars: int,
    history_chars: int,
) -> str:
    lines = [
        f"### `{case.get('state_id')}` rank={case.get('rank')} / {case.get('num_actions')}",
        "",
        f"- target type: `{case.get('target_action_type')}`",
        f"- target action: `{case.get('target_action')}`",
        f"- target score: `{fmt_float(case.get('target_score'))}`",
        f"- top1 action: `{case.get('top1_action')}`",
        f"- top1 type: `{case.get('top1_action_type')}`",
        f"- top1 score: `{fmt_float(case.get('top1_score'))}`",
        f"- top1-target margin: `{fmt_float(case.get('score_margin_top1_minus_target'))}`",
    ]
    if state:
        lines.extend(
            [
                f"- task: {state.get('instruction', '')}",
                "",
                "**Recent History**",
                "",
                format_history(state.get("history") or [], history_chars),
                "",
                "**Observation Prefix**",
                "",
                truncate(state.get("observation", ""), obs_chars),
            ]
        )
    lines.extend(["", "**Top-5 actions**", ""])
    for idx, item in enumerate(case.get("top5") or [], start=1):
        marker = " target" if item.get("is_target") else ""
        lines.append(
            f"{idx}. `{item.get('action')}` "
            f"type={item.get('action_type')} score={fmt_float(item.get('score'))}{marker}"
        )
    return "\n".join(lines)


def pick_cases(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return cases[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-json", default="data/processed/scorer_baseline/yesno_scorer_full_ranking.json")
    parser.add_argument("--states", default="data/processed/scorer_baseline/valid_states.jsonl")
    parser.add_argument("--out-md", default="reports/yesno_scorer_case_analysis.md")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--obs-chars", type=int, default=1200)
    parser.add_argument("--history-chars", type=int, default=400)
    args = parser.parse_args()

    ranking = load_json(Path(args.ranking_json))
    states = {row["state_id"]: row for row in load_jsonl(Path(args.states))}
    cases = ranking.get("case_records") or ranking.get("examples") or []
    if not cases:
        raise ValueError(
            "No case_records found. Re-run evaluate_yesno_scorer_ranking.py with the latest code."
        )

    successes = [case for case in cases if int(case["rank"]) == 1]
    near_misses = [case for case in cases if 1 < int(case["rank"]) <= 3]
    failures = [case for case in cases if int(case["rank"]) > 3]
    severe_failures = sorted(
        failures,
        key=lambda x: (
            int(x["rank"]),
            float(x.get("score_margin_top1_minus_target") or 0.0),
        ),
        reverse=True,
    )

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_type[str(case.get("target_action_type") or "unknown")].append(case)

    lines = [
        "# Yes/No Scorer Case Analysis",
        "",
        f"Ranking file: `{args.ranking_json}`",
        f"States file: `{args.states}`",
        "",
        "## Summary",
        "",
        f"- total cases: {len(cases)}",
        f"- top1 successes: {len(successes)}",
        f"- top3 near misses: {len(near_misses)}",
        f"- rank>3 failures: {len(failures)}",
        "",
        "## Cases By Target Type",
        "",
        "| Type | cases | top1 | top3 | failures(rank>3) |",
        "|---|---:|---:|---:|---:|",
    ]
    for action_type in sorted(by_type):
        rows = by_type[action_type]
        top1 = sum(int(case["rank"]) == 1 for case in rows)
        top3 = sum(int(case["rank"]) <= 3 for case in rows)
        fail = sum(int(case["rank"]) > 3 for case in rows)
        lines.append(f"| {action_type} | {len(rows)} | {top1} | {top3} | {fail} |")

    sections = [
        ("Top-1 Success Examples", pick_cases(successes, args.limit)),
        ("Top-3 Near Miss Examples", pick_cases(near_misses, args.limit)),
        ("Severe Failure Examples", pick_cases(severe_failures, args.limit)),
    ]
    for action_type in ["item_click", "click_other", "info", "navigation", "pagination", "buy"]:
        typed_failures = [case for case in failures if case.get("target_action_type") == action_type]
        if typed_failures:
            sections.append((f"{action_type} Failures", pick_cases(typed_failures, args.limit)))

    for title, rows in sections:
        lines.extend(["", f"## {title}", ""])
        if not rows:
            lines.append("No cases.")
            continue
        for case in rows:
            lines.append(format_case(case, states.get(case["state_id"]), args.obs_chars, args.history_chars))
            lines.append("")

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
