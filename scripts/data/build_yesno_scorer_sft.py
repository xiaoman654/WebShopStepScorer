#!/usr/bin/env python3
"""Convert scorer classification rows into Qwen-style Yes/No SFT data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_rows is not None and len(rows) >= max_rows:
                    break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate(text: Any, max_chars: int) -> str:
    value = str(text or "")
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars]


def format_history(history: list[dict[str, Any]], max_history_chars: int) -> str:
    if not history:
        return "None"

    chunks = []
    for item in history:
        step_id = item.get("step_id", "?")
        action = str(item.get("action", ""))
        obs = truncate(item.get("observation", ""), max_history_chars)
        chunks.append(f"Step {step_id} action: {action}\nStep {step_id} observation: {obs}")
    return "\n\n".join(chunks)


def make_user_prompt(row: dict[str, Any], max_observation_chars: int, max_history_chars: int) -> str:
    return (
        "You are judging whether a candidate WebShop action is a good next action.\n"
        "Use only the task, recent history, current observation, and candidate action.\n"
        "Answer with exactly one word: Yes or No.\n\n"
        f"Task:\n{row.get('instruction', '')}\n\n"
        f"Recent history:\n{format_history(row.get('history') or [], max_history_chars)}\n\n"
        f"Current observation:\n{truncate(row.get('observation', ''), max_observation_chars)}\n\n"
        f"Candidate action:\n{row.get('candidate_action', '')}\n\n"
        f"Candidate action type: {row.get('candidate_action_type', '')}\n\n"
        "Is this candidate action a good next action?"
    )


def convert_row(row: dict[str, Any], max_observation_chars: int, max_history_chars: int) -> dict[str, Any]:
    label = int(row["label"])
    answer = "Yes" if label == 1 else "No"
    return {
        "sample_id": row.get("sample_id"),
        "state_id": row.get("state_id"),
        "trajectory_id": row.get("trajectory_id"),
        "step_id": row.get("step_id"),
        "candidate_action": row.get("candidate_action"),
        "candidate_action_type": row.get("candidate_action_type"),
        "target_action": row.get("target_action"),
        "target_action_type": row.get("target_action_type"),
        "label": label,
        "answer": answer,
        "messages": [
            {"role": "user", "content": make_user_prompt(row, max_observation_chars, max_history_chars)},
            {"role": "assistant", "content": answer},
        ],
        "source": "webshop_step_scorer_yesno_sft",
    }


def convert_file(
    input_path: Path,
    output_path: Path,
    max_observation_chars: int,
    max_history_chars: int,
    max_rows: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(input_path, max_rows=max_rows)
    converted = [
        convert_row(row, max_observation_chars, max_history_chars)
        for row in rows
    ]
    write_jsonl(output_path, converted)

    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "num_rows": len(converted),
        "label_counts": dict(Counter(int(row["label"]) for row in converted).most_common()),
        "answer_counts": dict(Counter(str(row["answer"]) for row in converted).most_common()),
        "candidate_action_type_counts": dict(
            Counter(str(row.get("candidate_action_type") or "unknown") for row in converted).most_common()
        ),
        "max_observation_chars": max_observation_chars,
        "max_history_chars": max_history_chars,
        "max_rows": max_rows,
    }
    return converted, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/scorer_baseline/train.jsonl")
    parser.add_argument("--valid", default="data/processed/scorer_baseline/valid.jsonl")
    parser.add_argument("--out-dir", default="data/processed/yesno_scorer_sft")
    parser.add_argument("--max-observation-chars", type=int, default=4000)
    parser.add_argument("--max-history-chars", type=int, default=800)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-valid-rows", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, train_stats = convert_file(
        Path(args.train),
        out_dir / "train.jsonl",
        args.max_observation_chars,
        args.max_history_chars,
        args.max_train_rows,
    )
    _, valid_stats = convert_file(
        Path(args.valid),
        out_dir / "valid.jsonl",
        args.max_observation_chars,
        args.max_history_chars,
        args.max_valid_rows,
    )

    stats = {
        "train": train_stats,
        "valid": valid_stats,
        "format": "Qwen chat messages; assistant answer is exactly Yes or No",
        "note": "The prompt includes candidate_action_type but does not include target_action_type.",
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote: {out_dir / 'train.jsonl'}")
    print(f"wrote: {out_dir / 'valid.jsonl'}")
    print(f"wrote: {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
