#!/usr/bin/env python3
"""Simulate using the scorer as an offline WebShop action selector."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def action_type_counts(cases: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(case.get(key) or "unknown") for case in cases).items()))


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(cases)
    exact_top1 = sum(int(case.get("rank") or 0) == 1 for case in cases)
    target_in_top3 = sum(int(case.get("rank") or 0) <= 3 for case in cases)
    target_in_top5 = sum(int(case.get("rank") or 0) <= 5 for case in cases)
    same_type_top1 = sum(case.get("top1_action_type") == case.get("target_action_type") for case in cases)
    same_type_miss = sum(
        int(case.get("rank") or 0) != 1
        and case.get("top1_action_type") == case.get("target_action_type")
        for case in cases
    )
    margins = [float(case.get("score_margin_top1_minus_target") or 0.0) for case in cases]
    close_failures = sum(
        int(case.get("rank") or 0) > 3
        and float(case.get("score_margin_top1_minus_target") or 0.0) <= 0.05
        for case in cases
    )
    return {
        "num_states": n,
        "exact_top1_match": safe_div(exact_top1, n),
        "exact_top1_count": exact_top1,
        "same_type_top1_match": safe_div(same_type_top1, n),
        "same_type_top1_count": same_type_top1,
        "same_type_non_exact_count": same_type_miss,
        "target_in_top3": safe_div(target_in_top3, n),
        "target_in_top3_count": target_in_top3,
        "target_in_top5": safe_div(target_in_top5, n),
        "target_in_top5_count": target_in_top5,
        "close_rank_gt3_count": close_failures,
        "avg_top1_target_margin": sum(margins) / n if n else 0.0,
        "mean_rank": sum(int(case.get("rank") or 0) for case in cases) / n if n else 0.0,
    }


def topk_type_recoverability(cases: list[dict[str, Any]], k: int) -> dict[str, Any]:
    recovered = 0
    target_type_present = 0
    for case in cases:
        target = case.get("target_action")
        target_type = case.get("target_action_type")
        topk = case.get("scored_actions", [])[:k] or case.get("top5", [])[:k]
        if any(item.get("action") == target for item in topk):
            recovered += 1
        if any(item.get("action_type") == target_type for item in topk):
            target_type_present += 1
    n = len(cases)
    return {
        "k": k,
        "target_action_in_topk": safe_div(recovered, n),
        "target_action_in_topk_count": recovered,
        "target_type_in_topk": safe_div(target_type_present, n),
        "target_type_in_topk_count": target_type_present,
    }


def confusion_matrix(cases: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        target_type = str(case.get("target_action_type") or "unknown")
        top1_type = str(case.get("top1_action_type") or "unknown")
        matrix[target_type][top1_type] += 1
    return {row: dict(sorted(cols.items())) for row, cols in sorted(matrix.items())}


def by_target_type(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[str(case.get("target_action_type") or "unknown")].append(case)
    return {key: summarize_cases(rows) for key, rows in sorted(groups.items())}


def best_action_in_type(case: dict[str, Any], action_type: str) -> dict[str, Any] | None:
    candidates = [item for item in case.get("scored_actions", []) if item.get("action_type") == action_type]
    return candidates[0] if candidates else None


def selected_action_with_buy_penalty(case: dict[str, Any], penalty: float) -> dict[str, Any] | None:
    scored = []
    for item in case.get("scored_actions", []):
        adjusted = float(item.get("score") or 0.0)
        if item.get("action_type") == "buy":
            adjusted -= penalty
        scored.append((adjusted, str(item.get("action") or ""), item))
    if not scored:
        return None
    return sorted(scored, key=lambda x: (-x[0], x[1]))[0][2]


def selected_action_conservative_buy(case: dict[str, Any], margin: float) -> dict[str, Any] | None:
    actions = case.get("scored_actions", [])
    if not actions:
        return None
    top1 = actions[0]
    if top1.get("action_type") != "buy" or len(actions) == 1:
        return top1
    second = actions[1]
    gap = float(top1.get("score") or 0.0) - float(second.get("score") or 0.0)
    if gap <= margin:
        return second
    return top1


def summarize_policy_selection(cases: list[dict[str, Any]], selections: list[dict[str, Any] | None]) -> dict[str, Any]:
    n = len(cases)
    exact = 0
    same_type = 0
    selected_types = Counter()
    by_type: dict[str, list[tuple[dict[str, Any], dict[str, Any] | None]]] = defaultdict(list)
    for case, selected in zip(cases, selections):
        target_type = str(case.get("target_action_type") or "unknown")
        by_type[target_type].append((case, selected))
        if selected is None:
            selected_types["none"] += 1
            continue
        selected_types[str(selected.get("action_type") or "unknown")] += 1
        if selected.get("action") == case.get("target_action"):
            exact += 1
        if selected.get("action_type") == case.get("target_action_type"):
            same_type += 1

    by_target = {}
    for target_type, rows in sorted(by_type.items()):
        row_n = len(rows)
        row_exact = sum(
            selected is not None and selected.get("action") == case.get("target_action")
            for case, selected in rows
        )
        row_same = sum(
            selected is not None and selected.get("action_type") == case.get("target_action_type")
            for case, selected in rows
        )
        by_target[target_type] = {
            "num_states": row_n,
            "exact_match": safe_div(row_exact, row_n),
            "exact_count": row_exact,
            "same_type_match": safe_div(row_same, row_n),
            "same_type_count": row_same,
        }
    return {
        "num_states": n,
        "exact_match": safe_div(exact, n),
        "exact_count": exact,
        "same_type_match": safe_div(same_type, n),
        "same_type_count": same_type,
        "selected_type_distribution": dict(sorted(selected_types.items())),
        "by_target_action_type": by_target,
    }


def policy_simulations(cases: list[dict[str, Any]]) -> dict[str, Any]:
    sims = {
        "greedy_top1": [case.get("scored_actions", [None])[0] for case in cases],
        "oracle_target_type": [
            best_action_in_type(case, str(case.get("target_action_type") or "unknown")) for case in cases
        ],
    }
    for penalty in [0.05, 0.10, 0.20]:
        sims[f"buy_penalty_{penalty:.2f}"] = [selected_action_with_buy_penalty(case, penalty) for case in cases]
    for margin in [0.05, 0.10, 0.20]:
        sims[f"conservative_buy_margin_{margin:.2f}"] = [
            selected_action_conservative_buy(case, margin) for case in cases
        ]
    return {name: summarize_policy_selection(cases, selections) for name, selections in sims.items()}


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    lines = [
        "# Selector Simulation",
        "",
        f"Ranking file: `{summary['ranking_json']}`",
        "",
        "This report simulates greedy scorer selection without environment rollout.",
        "It measures whether scorer top-1 matches the human demonstration action,",
        "whether mistakes stay within the same action type, and whether top-k policies could recover the target.",
        "",
        "## Overall",
        "",
        "| Metric | Value | Count |",
        "|---|---:|---:|",
        f"| exact top1 match | {overall['exact_top1_match']:.4f} | {overall['exact_top1_count']} |",
        f"| same-type top1 match | {overall['same_type_top1_match']:.4f} | {overall['same_type_top1_count']} |",
        f"| same-type but non-exact | N/A | {overall['same_type_non_exact_count']} |",
        f"| target in top3 | {overall['target_in_top3']:.4f} | {overall['target_in_top3_count']} |",
        f"| target in top5 | {overall['target_in_top5']:.4f} | {overall['target_in_top5_count']} |",
        f"| close rank>3, margin <= 0.05 | N/A | {overall['close_rank_gt3_count']} |",
        f"| avg top1-target margin | {overall['avg_top1_target_margin']:.6f} | N/A |",
        f"| mean rank | {overall['mean_rank']:.4f} | N/A |",
        "",
        "## Top-k Recoverability",
        "",
        "| k | target action in top-k | count | target type in top-k | count |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in summary["topk_recoverability"]:
        lines.append(
            f"| {item['k']} | {item['target_action_in_topk']:.4f} | "
            f"{item['target_action_in_topk_count']} | {item['target_type_in_topk']:.4f} | "
            f"{item['target_type_in_topk_count']} |"
        )

    lines.extend(
        [
            "",
            "## Top1 Action Type Distribution",
            "",
            "| Top1 type | count | share |",
            "|---|---:|---:|",
        ]
    )
    total = overall["num_states"]
    for action_type, count in summary["top1_action_type_distribution"].items():
        lines.append(f"| {action_type} | {count} | {safe_div(count, total):.4f} |")

    lines.extend(
        [
            "",
            "## By Target Action Type",
            "",
            "| Target type | states | exact top1 | same-type top1 | target top3 | mean rank | avg margin |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for action_type, values in summary["by_target_action_type"].items():
        lines.append(
            f"| {action_type} | {values['num_states']} | {values['exact_top1_match']:.4f} | "
            f"{values['same_type_top1_match']:.4f} | {values['target_in_top3']:.4f} | "
            f"{values['mean_rank']:.3f} | {values['avg_top1_target_margin']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Action Type Confusion Matrix",
            "",
            "Rows are target action types; columns are scorer top1 action types.",
            "",
        ]
    )
    all_cols = sorted({col for row in summary["confusion_matrix"].values() for col in row})
    lines.append("| target \\ top1 | " + " | ".join(all_cols) + " |")
    lines.append("|---|" + "|".join("---:" for _ in all_cols) + "|")
    for row_type, row in summary["confusion_matrix"].items():
        values = [str(row.get(col, 0)) for col in all_cols]
        lines.append(f"| {row_type} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Policy Simulations",
            "",
            "These are offline diagnostics, not environment rollouts.",
            "`oracle_target_type` measures the upper bound of a perfect action-type gate plus current within-type scorer ranking.",
            "",
            "| Policy | exact match | same-type match | buy top1 share | item_click top1 share | navigation top1 share | pagination top1 share |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy_name, values in summary["policy_simulations"].items():
        dist = values["selected_type_distribution"]
        total_selected = sum(dist.values())
        lines.append(
            f"| {policy_name} | {values['exact_match']:.4f} | {values['same_type_match']:.4f} | "
            f"{safe_div(dist.get('buy', 0), total_selected):.4f} | "
            f"{safe_div(dist.get('item_click', 0), total_selected):.4f} | "
            f"{safe_div(dist.get('navigation', 0), total_selected):.4f} | "
            f"{safe_div(dist.get('pagination', 0), total_selected):.4f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-json", default="data/processed/scorer_baseline/yesno_scorer_full_ranking.json")
    parser.add_argument("--out-json", default="reports/selector_simulation.json")
    parser.add_argument("--out-md", default="reports/selector_simulation.md")
    parser.add_argument("--topk", default="1,2,3,5")
    args = parser.parse_args()

    ranking = load_json(Path(args.ranking_json))
    cases = ranking.get("case_records") or []
    if not cases:
        raise ValueError("No case_records found in ranking JSON. Re-run ranking with latest evaluator.")

    topk_values = [int(item.strip()) for item in args.topk.split(",") if item.strip()]
    summary = {
        "ranking_json": args.ranking_json,
        "overall": summarize_cases(cases),
        "topk_recoverability": [topk_type_recoverability(cases, k) for k in topk_values],
        "target_action_type_distribution": action_type_counts(cases, "target_action_type"),
        "top1_action_type_distribution": action_type_counts(cases, "top1_action_type"),
        "by_target_action_type": by_target_type(cases),
        "confusion_matrix": confusion_matrix(cases),
        "policy_simulations": policy_simulations(cases),
    }
    write_json(Path(args.out_json), summary)
    write_markdown(Path(args.out_md), summary)
    print(f"wrote: {args.out_json}")
    print(f"wrote: {args.out_md}")


if __name__ == "__main__":
    main()
