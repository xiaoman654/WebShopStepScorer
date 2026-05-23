#!/usr/bin/env python3
"""Train a CPU TF-IDF action scorer and evaluate offline ranking metrics."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def truncate(text: Any, max_chars: int) -> str:
    value = str(text or "")
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars]


def format_history(history: list[dict[str, Any]], max_history_chars: int) -> str:
    chunks = []
    for item in history:
        action = str(item.get("action", ""))
        obs = truncate(item.get("observation", ""), max_history_chars)
        chunks.append(f"Previous action: {action}\nPrevious observation: {obs}")
    return "\n".join(chunks) if chunks else "None"


def format_example(row: dict[str, Any], max_observation_chars: int, max_history_chars: int) -> str:
    return (
        f"Task:\n{row.get('instruction', '')}\n\n"
        f"History:\n{format_history(row.get('history') or [], max_history_chars)}\n\n"
        f"Current observation:\n{truncate(row.get('observation', ''), max_observation_chars)}\n\n"
        f"Candidate action:\n{row.get('candidate_action', '')}\n\n"
        f"Candidate action type: {row.get('candidate_action_type', '')}\n"
        f"Target action type hint: {row.get('target_action_type', '')}\n"
    )


def format_state_action(
    state: dict[str, Any],
    action: str,
    action_type: str,
    max_observation_chars: int,
    max_history_chars: int,
) -> str:
    return (
        f"Task:\n{state.get('instruction', '')}\n\n"
        f"History:\n{format_history(state.get('history') or [], max_history_chars)}\n\n"
        f"Current observation:\n{truncate(state.get('observation', ''), max_observation_chars)}\n\n"
        f"Candidate action:\n{action}\n\n"
        f"Candidate action type: {action_type}\n"
        f"Target action type hint: {state.get('target_action_type', '')}\n"
    )


def action_type_for_state_action(state: dict[str, Any], action: str) -> str:
    actions = state.get("available_actions") or []
    types = state.get("available_action_types") or []
    for idx, candidate in enumerate(actions):
        if candidate == action and idx < len(types):
            return str(types[idx])
    return "unknown"


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
        "top1": sum(rank == 1 for rank in ranks) / n,
        "top3": sum(rank <= 3 for rank in ranks) / n,
        "mrr": sum(1.0 / rank for rank in ranks) / n,
        "mean_rank": sum(ranks) / n,
        "mean_num_actions": sum(len(s.get("available_actions") or []) for s in states) / n,
    }


def evaluate_ranking(
    model: Any,
    states: list[dict[str, Any]],
    max_observation_chars: int,
    max_history_chars: int,
) -> dict[str, Any]:
    ranks = []
    by_type_states: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type_ranks: dict[str, list[int]] = defaultdict(list)

    for state in states:
        actions = list(state.get("available_actions") or [])
        texts = [
            format_state_action(
                state,
                action,
                action_type_for_state_action(state, action),
                max_observation_chars,
                max_history_chars,
            )
            for action in actions
        ]
        scores = model.predict_proba(texts)[:, 1]
        scored = sorted(zip(actions, scores), key=lambda item: (-item[1], item[0]))
        rank = next(idx for idx, (action, _) in enumerate(scored, start=1) if action == state["target_action"])
        ranks.append(rank)

        target_type = str(state.get("target_action_type") or "unknown")
        by_type_states[target_type].append(state)
        by_type_ranks[target_type].append(rank)

    return {
        "overall": summarize_ranks(states, ranks),
        "by_target_action_type": {
            key: summarize_ranks(by_type_states[key], by_type_ranks[key])
            for key in sorted(by_type_states)
        },
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# TF-IDF Scorer Baseline",
        "",
        f"Train examples: {summary['num_train_examples']}",
        f"Valid examples: {summary['num_valid_examples']}",
        f"Valid states: {summary['num_valid_states']}",
        "",
        "## Classification",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| AUC | {summary['classification']['auc']:.4f} |",
        "",
        "## Ranking",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    overall = summary["ranking"]["overall"]
    for key in ["top1", "top3", "mrr", "mean_rank", "mean_num_actions"]:
        lines.append(f"| {key} | {overall[key]:.4f} |")

    lines.extend(
        [
            "",
            "## Ranking By Target Action Type",
            "",
            "| Type | states | top1 | top3 | mrr | mean_rank | mean_actions |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for action_type, values in summary["ranking"]["by_target_action_type"].items():
        lines.append(
            "| {action_type} | {num_states} | {top1:.4f} | {top3:.4f} | "
            "{mrr:.4f} | {mean_rank:.3f} | {mean_num_actions:.3f} |".format(
                action_type=action_type,
                **values,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/scorer_baseline/train.jsonl")
    parser.add_argument("--valid", default="data/processed/scorer_baseline/valid.jsonl")
    parser.add_argument("--valid-states", default="data/processed/scorer_baseline/valid_states.jsonl")
    parser.add_argument("--out-json", default="data/processed/scorer_baseline/tfidf_scorer_eval.json")
    parser.add_argument("--out-md", default="reports/tfidf_scorer_baseline.md")
    parser.add_argument("--model-out", default="outputs/tfidf_scorer/model.pkl")
    parser.add_argument("--max-features", type=int, default=200000)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--max-observation-chars", type=int, default=4000)
    parser.add_argument("--max-history-chars", type=int, default=800)
    args = parser.parse_args()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is required for this script. Install it with: "
            "pip install scikit-learn"
        ) from exc

    train_rows = load_jsonl(Path(args.train))
    valid_rows = load_jsonl(Path(args.valid))
    valid_states = load_jsonl(Path(args.valid_states))

    x_train = [
        format_example(row, args.max_observation_chars, args.max_history_chars)
        for row in train_rows
    ]
    y_train = [int(row["label"]) for row in train_rows]
    x_valid = [
        format_example(row, args.max_observation_chars, args.max_history_chars)
        for row in valid_rows
    ]
    y_valid = [int(row["label"]) for row in valid_rows]

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=args.max_features,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=args.max_iter,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    valid_scores = model.predict_proba(x_valid)[:, 1]
    auc = roc_auc_score(y_valid, valid_scores)
    ranking = evaluate_ranking(
        model,
        valid_states,
        args.max_observation_chars,
        args.max_history_chars,
    )

    summary = {
        "train_file": args.train,
        "valid_file": args.valid,
        "valid_states_file": args.valid_states,
        "num_train_examples": len(train_rows),
        "num_valid_examples": len(valid_rows),
        "num_valid_states": len(valid_states),
        "max_observation_chars": args.max_observation_chars,
        "max_history_chars": args.max_history_chars,
        "classification": {"auc": auc},
        "ranking": ranking,
    }

    write_json(Path(args.out_json), summary)
    write_markdown(Path(args.out_md), summary)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.model_out).open("wb") as f:
        pickle.dump(model, f)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote: {args.out_json}")
    print(f"wrote: {args.out_md}")
    print(f"wrote: {args.model_out}")


if __name__ == "__main__":
    main()
