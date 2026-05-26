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


def history_chars(state: dict[str, Any] | None) -> int:
    if not state:
        return 0
    return sum(len(str(item.get("observation", ""))) + len(str(item.get("action", ""))) for item in state.get("history") or [])


def observation_chars(state: dict[str, Any] | None) -> int:
    if not state:
        return 0
    return len(str(state.get("observation", "")))


def action_index(case: dict[str, Any], action: str | None) -> int | None:
    if not action:
        return None
    for item in case.get("scored_actions") or []:
        if item.get("action") == action:
            return int(item.get("rank", 0)) - 1
    return None


def classify_error(case: dict[str, Any], state: dict[str, Any] | None) -> str:
    rank = int(case.get("rank") or 0)
    if rank == 1:
        return "top1_success"

    target_type = str(case.get("target_action_type") or "unknown")
    top1_type = str(case.get("top1_action_type") or "unknown")
    top1_action = str(case.get("top1_action") or "").lower()
    target_index = case.get("target_action_index")
    top1_index = action_index(case, case.get("top1_action"))

    if top1_type != target_type:
        if top1_type == "buy":
            return "late_stage_buy_bias"
        if top1_type == "info":
            return "generic_info_bias"
        return "type_confusion"

    if target_type == "item_click":
        return "within_type_item_confusion"
    if target_type in {"info", "navigation", "pagination"}:
        return f"within_type_{target_type}_confusion"
    if "description" in top1_action or "feature" in top1_action or "review" in top1_action:
        return "generic_info_bias"
    if isinstance(target_index, int) and isinstance(top1_index, int) and top1_index < target_index:
        return "position_bias"
    if state and target_type in {"item_click", "click_other", "option"}:
        return "attribute_or_entity_mismatch"
    return "other_failure"


def format_case(
    case: dict[str, Any],
    state: dict[str, Any] | None,
    obs_chars: int,
    max_history_chars: int,
) -> str:
    lines = [
        f"### `{case.get('state_id')}` rank={case.get('rank')} / {case.get('num_actions')}",
        "",
        f"- error class: `{case.get('error_class', 'unknown')}`",
        f"- target type: `{case.get('target_action_type')}`",
        f"- target action: `{case.get('target_action')}`",
        f"- target action index: `{case.get('target_action_index')}`",
        f"- target score: `{fmt_float(case.get('target_score'))}`",
        f"- top1 action: `{case.get('top1_action')}`",
        f"- top1 type: `{case.get('top1_action_type')}`",
        f"- top1 score: `{fmt_float(case.get('top1_score'))}`",
        f"- top1-target margin: `{fmt_float(case.get('score_margin_top1_minus_target'))}`",
        f"- top1 same type as target: `{case.get('top1_same_type')}`",
    ]
    if state:
        lines.extend(
            [
                f"- observation chars: `{observation_chars(state)}`",
                f"- history chars: `{history_chars(state)}`",
                f"- task: {state.get('instruction', '')}",
                "",
                "**Recent History**",
                "",
                format_history(state.get("history") or [], max_history_chars),
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


def format_compact_case(case: dict[str, Any]) -> str:
    top5 = case.get("top5") or []
    top3 = [
        f"{idx}. {item.get('action')} ({item.get('action_type')}, {fmt_float(item.get('score'))})"
        for idx, item in enumerate(top5[:3], start=1)
    ]
    return "\n".join(
        [
            f"- `{case.get('state_id')}` | `{case.get('error_class')}` | "
            f"rank {case.get('rank')}/{case.get('num_actions')} | "
            f"target `{case.get('target_action_type')}` score {fmt_float(case.get('target_score'))} | "
            f"top1 `{case.get('top1_action_type')}` score {fmt_float(case.get('top1_score'))} | "
            f"margin {fmt_float(case.get('score_margin_top1_minus_target'))}",
            f"  - target: `{case.get('target_action')}`",
            f"  - top3: {' ; '.join(top3)}",
        ]
    )


def pick_cases(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return cases[:limit]


def write_compact_report(
    path: Path,
    ranking_json: str,
    states_path: str,
    cases: list[dict[str, Any]],
    by_type: dict[str, list[dict[str, Any]]],
    by_error: dict[str, list[dict[str, Any]]],
    limit: int,
) -> None:
    successes = [case for case in cases if int(case["rank"]) == 1]
    near_misses = [case for case in cases if 1 < int(case["rank"]) <= 3]
    failures = [case for case in cases if int(case["rank"]) > 3]
    same_type_misses = [case for case in failures if case.get("top1_same_type")]
    cross_type_misses = [case for case in failures if not case.get("top1_same_type")]
    close_misses = [case for case in failures if float(case.get("score_margin_top1_minus_target") or 0.0) <= 0.05]
    margins = [float(case.get("score_margin_top1_minus_target") or 0.0) for case in cases]
    severe_failures = sorted(
        failures,
        key=lambda x: (
            int(x["rank"]),
            float(x.get("score_margin_top1_minus_target") or 0.0),
        ),
        reverse=True,
    )

    lines = [
        "# Yes/No Scorer Case Analysis Compact",
        "",
        f"Ranking file: `{ranking_json}`",
        f"States file: `{states_path}`",
        "",
        "## Key Takeaways",
        "",
        f"- total states: {len(cases)}",
        f"- top1 successes: {len(successes)} ({len(successes) / len(cases):.4f})",
        f"- top3 coverage: {sum(int(case['rank']) <= 3 for case in cases)} ({sum(int(case['rank']) <= 3 for case in cases) / len(cases):.4f})",
        f"- rank>3 failures: {len(failures)} ({len(failures) / len(cases):.4f})",
        f"- same-type rank>3 failures: {len(same_type_misses)}",
        f"- cross-type rank>3 failures: {len(cross_type_misses)}",
        f"- close rank>3 failures, margin <= 0.05: {len(close_misses)}",
        f"- average top1-target margin: {sum(margins) / len(margins):.6f}",
        "",
        "## Target Action Type Summary",
        "",
        "| Type | cases | top1 | top3 | rank>3 | avg margin |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for action_type in sorted(by_type):
        rows = by_type[action_type]
        top1 = sum(int(case["rank"]) == 1 for case in rows)
        top3 = sum(int(case["rank"]) <= 3 for case in rows)
        fail = sum(int(case["rank"]) > 3 for case in rows)
        avg_margin = sum(float(case.get("score_margin_top1_minus_target") or 0.0) for case in rows) / len(rows)
        lines.append(f"| {action_type} | {len(rows)} | {top1} | {top3} | {fail} | {avg_margin:.6f} |")

    lines.extend(
        [
            "",
            "## Error Taxonomy Summary",
            "",
            "| Error class | cases | top1 | top3 | rank>3 | avg margin |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for error_class in sorted(by_error):
        rows = by_error[error_class]
        top1 = sum(int(case["rank"]) == 1 for case in rows)
        top3 = sum(int(case["rank"]) <= 3 for case in rows)
        fail = sum(int(case["rank"]) > 3 for case in rows)
        avg_margin = sum(float(case.get("score_margin_top1_minus_target") or 0.0) for case in rows) / len(rows)
        lines.append(f"| {error_class} | {len(rows)} | {top1} | {top3} | {fail} | {avg_margin:.6f} |")

    lines.extend(
        [
            "",
            "## Minimal Representative Cases",
            "",
            "### Severe Failures",
            "",
        ]
    )
    for case in severe_failures[:limit]:
        lines.append(format_compact_case(case))
        lines.append("")

    lines.extend(["### Near Misses", ""])
    for case in near_misses[:limit]:
        lines.append(format_compact_case(case))
        lines.append("")

    lines.extend(["### Successes", ""])
    for case in successes[:limit]:
        lines.append(format_compact_case(case))
        lines.append("")

    for error_class in [
        "type_confusion",
        "within_type_item_confusion",
        "generic_info_bias",
        "late_stage_buy_bias",
        "position_bias",
        "attribute_or_entity_mismatch",
    ]:
        rows = [case for case in failures if case.get("error_class") == error_class]
        if not rows:
            continue
        lines.extend([f"### {error_class}", ""])
        for case in rows[:limit]:
            lines.append(format_compact_case(case))
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-json", default="data/processed/scorer_baseline/yesno_scorer_full_ranking.json")
    parser.add_argument("--states", default="data/processed/scorer_baseline/valid_states.jsonl")
    parser.add_argument("--out-md", default="reports/yesno_scorer_case_analysis.md")
    parser.add_argument("--compact-out-md", default="reports/yesno_scorer_case_analysis_compact.md")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--compact-limit", type=int, default=2)
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

    enriched_cases = []
    for case in cases:
        state = states.get(case["state_id"])
        enriched = dict(case)
        enriched["top1_same_type"] = case.get("top1_action_type") == case.get("target_action_type")
        enriched["observation_chars"] = observation_chars(state)
        enriched["history_chars"] = history_chars(state)
        enriched["error_class"] = classify_error(enriched, state)
        enriched_cases.append(enriched)
    cases = enriched_cases

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
    by_error: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_type[str(case.get("target_action_type") or "unknown")].append(case)
        by_error[str(case.get("error_class") or "unknown")].append(case)

    same_type_misses = [case for case in failures if case.get("top1_same_type")]
    cross_type_misses = [case for case in failures if not case.get("top1_same_type")]
    margins = [float(case.get("score_margin_top1_minus_target") or 0.0) for case in cases]
    close_misses = [case for case in failures if float(case.get("score_margin_top1_minus_target") or 0.0) <= 0.05]

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
        f"- same-type rank>3 failures: {len(same_type_misses)}",
        f"- cross-type rank>3 failures: {len(cross_type_misses)}",
        f"- close rank>3 failures (margin <= 0.05): {len(close_misses)}",
        f"- average top1-target margin: {sum(margins) / len(margins):.6f}",
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

    lines.extend(
        [
            "",
            "## Error Taxonomy",
            "",
            "| Error class | cases | top1 | top3 | rank>3 | avg margin |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for error_class in sorted(by_error):
        rows = by_error[error_class]
        top1 = sum(int(case["rank"]) == 1 for case in rows)
        top3 = sum(int(case["rank"]) <= 3 for case in rows)
        fail = sum(int(case["rank"]) > 3 for case in rows)
        avg_margin = sum(float(case.get("score_margin_top1_minus_target") or 0.0) for case in rows) / len(rows)
        lines.append(f"| {error_class} | {len(rows)} | {top1} | {top3} | {fail} | {avg_margin:.6f} |")

    lines.extend(
        [
            "",
            "## Length Buckets",
            "",
            "| Observation chars | cases | top1 | top3 | mean rank |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    buckets = [
        ("<=1k", lambda x: x <= 1000),
        ("1k-3k", lambda x: 1000 < x <= 3000),
        (">3k", lambda x: x > 3000),
    ]
    for name, pred in buckets:
        rows = [case for case in cases if pred(int(case.get("observation_chars") or 0))]
        if not rows:
            continue
        top1 = sum(int(case["rank"]) == 1 for case in rows) / len(rows)
        top3 = sum(int(case["rank"]) <= 3 for case in rows) / len(rows)
        mean_rank = sum(int(case["rank"]) for case in rows) / len(rows)
        lines.append(f"| {name} | {len(rows)} | {top1:.4f} | {top3:.4f} | {mean_rank:.3f} |")

    sections = [
        ("Top-1 Success Examples", pick_cases(successes, args.limit)),
        ("Top-3 Near Miss Examples", pick_cases(near_misses, args.limit)),
        ("Severe Failure Examples", pick_cases(severe_failures, args.limit)),
    ]
    for action_type in ["item_click", "click_other", "info", "navigation", "pagination", "buy"]:
        typed_failures = [case for case in failures if case.get("target_action_type") == action_type]
        if typed_failures:
            sections.append((f"{action_type} Failures", pick_cases(typed_failures, args.limit)))
    for error_class in [
        "type_confusion",
        "within_type_item_confusion",
        "generic_info_bias",
        "late_stage_buy_bias",
        "position_bias",
        "attribute_or_entity_mismatch",
    ]:
        error_rows = [case for case in failures if case.get("error_class") == error_class]
        if error_rows:
            sections.append((f"{error_class} Examples", pick_cases(error_rows, args.limit)))

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
    write_compact_report(
        Path(args.compact_out_md),
        args.ranking_json,
        args.states,
        cases,
        by_type,
        by_error,
        args.compact_limit,
    )


if __name__ == "__main__":
    main()
