#!/usr/bin/env python3
"""Evaluate non-model ranking baselines for WebShop action scorer states."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def action_type(action: str, state: dict[str, Any]) -> str:
    actions = state.get("available_actions") or []
    types = state.get("available_action_types") or []
    for idx, candidate in enumerate(actions):
        if candidate == action and idx < len(types):
            return str(types[idx])
    return "unknown"


def rank_of_target(
    state: dict[str, Any],
    scorer: Callable[[str, dict[str, Any]], float],
) -> int:
    target = state["target_action"]
    actions = list(state.get("available_actions") or [])
    scored = [(action, scorer(action, state)) for action in actions]
    scored.sort(key=lambda item: (-item[1], item[0]))
    for idx, (action, _) in enumerate(scored, start=1):
        if action == target:
            return idx
    raise ValueError(f"target action missing from state: {state.get('state_id')}")


def summarize_ranks(states: list[dict[str, Any]], ranks: list[int]) -> dict[str, Any]:
    n = len(ranks)
    if n == 0:
        return {
            "num_states": 0,
            "top1": 0.0,
            "top3": 0.0,
            "mrr": 0.0,
            "mean_rank": 0.0,
            "mean_num_actions": 0.0,
        }
    return {
        "num_states": n,
        "top1": sum(1 for rank in ranks if rank == 1) / n,
        "top3": sum(1 for rank in ranks if rank <= 3) / n,
        "mrr": sum(1.0 / rank for rank in ranks) / n,
        "mean_rank": sum(ranks) / n,
        "mean_num_actions": sum(len(s.get("available_actions") or []) for s in states) / n,
    }


def evaluate_baseline(
    states: list[dict[str, Any]],
    scorer: Callable[[str, dict[str, Any]], float],
) -> dict[str, Any]:
    ranks = [rank_of_target(state, scorer) for state in states]
    overall = summarize_ranks(states, ranks)

    by_type_states: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type_ranks: dict[str, list[int]] = defaultdict(list)
    for state, rank in zip(states, ranks):
        target_type = str(state.get("target_action_type") or "unknown")
        by_type_states[target_type].append(state)
        by_type_ranks[target_type].append(rank)

    by_type = {
        target_type: summarize_ranks(by_type_states[target_type], by_type_ranks[target_type])
        for target_type in sorted(by_type_states)
    }
    return {"overall": overall, "by_target_action_type": by_type}


def expected_random_baseline(states: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_states: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        grouped_states[str(state.get("target_action_type") or "unknown")].append(state)

    def expected_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return summarize_ranks([], [])
        n = len(rows)
        return {
            "num_states": n,
            "top1": sum(1 / len(row["available_actions"]) for row in rows) / n,
            "top3": sum(min(3, len(row["available_actions"])) / len(row["available_actions"]) for row in rows) / n,
            "mrr": sum(
                sum(1.0 / rank for rank in range(1, len(row["available_actions"]) + 1))
                / len(row["available_actions"])
                for row in rows
            )
            / n,
            "mean_rank": sum((len(row["available_actions"]) + 1) / 2 for row in rows) / n,
            "mean_num_actions": sum(len(row["available_actions"]) for row in rows) / n,
        }

    return {
        "overall": expected_summary(states),
        "by_target_action_type": {
            key: expected_summary(rows) for key, rows in sorted(grouped_states.items())
        },
    }


def markdown_table(title: str, rows: dict[str, dict[str, Any]]) -> str:
    lines = [
        f"## {title}",
        "",
        "| Group | states | top1 | top3 | mrr | mean_rank | mean_actions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, values in rows.items():
        lines.append(
            "| {group} | {num_states} | {top1:.4f} | {top3:.4f} | {mrr:.4f} | "
            "{mean_rank:.3f} | {mean_num_actions:.3f} |".format(group=group, **values)
        )
    return "\n".join(lines)


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Offline Ranking Baselines",
        "",
        f"State file: `{summary['state_file']}`",
        f"Number of states: {summary['num_states']}",
        "",
        "These are non-model baselines for the discrete admissible-action ranking task.",
        "A learned scorer should clearly beat these baselines before being considered for RL integration.",
        "",
        "| Baseline | top1 | top3 | mrr | mean_rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in summary["baselines"].items():
        overall = result["overall"]
        lines.append(
            f"| {name} | {overall['top1']:.4f} | {overall['top3']:.4f} | "
            f"{overall['mrr']:.4f} | {overall['mean_rank']:.3f} |"
        )
    lines.extend(["", markdown_table("By Target Action Type: random_expected", summary["baselines"]["random_expected"]["by_target_action_type"])])
    lines.extend(["", markdown_table("By Target Action Type: action_type_prior", summary["baselines"]["action_type_prior"]["by_target_action_type"])])
    lines.extend(["", markdown_table("By Target Action Type: heuristic_action_order", summary["baselines"]["heuristic_action_order"]["by_target_action_type"])])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", default="data/processed/scorer_baseline/valid_states.jsonl")
    parser.add_argument("--train-states", default="data/processed/scorer_baseline/train_states.jsonl")
    parser.add_argument("--out-json", default="data/processed/scorer_baseline/ranking_baselines.json")
    parser.add_argument("--out-md", default="reports/offline_ranking_baselines.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    states = load_jsonl(Path(args.states))
    train_states = load_jsonl(Path(args.train_states)) if Path(args.train_states).exists() else states
    rng = random.Random(args.seed)

    type_prior = Counter(str(row.get("target_action_type") or "unknown") for row in train_states)
    action_prior = Counter(str(row.get("target_action") or "") for row in train_states)
    heuristic_order = {
        "buy": 100,
        "item_click": 90,
        "option": 80,
        "info": 70,
        "pagination": 60,
        "navigation": 50,
        "click_other": 40,
        "search": 30,
        "other": 0,
        "unknown": 0,
    }

    random_scores = {
        (state["state_id"], action): rng.random()
        for state in states
        for action in state.get("available_actions") or []
    }

    baselines = {
        "random_expected": expected_random_baseline(states),
        "random_sampled": evaluate_baseline(
            states,
            lambda action, state: random_scores[(state["state_id"], action)],
        ),
        "action_type_prior": evaluate_baseline(
            states,
            lambda action, state: float(type_prior[action_type(action, state)]),
        ),
        "action_text_prior": evaluate_baseline(
            states,
            lambda action, state: float(action_prior[action]),
        ),
        "heuristic_action_order": evaluate_baseline(
            states,
            lambda action, state: float(heuristic_order.get(action_type(action, state), 0)),
        ),
    }

    summary = {
        "state_file": args.states,
        "train_state_file": args.train_states,
        "num_states": len(states),
        "seed": args.seed,
        "baselines": baselines,
    }
    write_json(Path(args.out_json), summary)
    write_markdown_report(Path(args.out_md), summary)
    print(json.dumps(summary["baselines"], indent=2, ensure_ascii=False))
    print(f"wrote: {args.out_json}")
    print(f"wrote: {args.out_md}")


if __name__ == "__main__":
    main()
