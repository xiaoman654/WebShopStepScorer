#!/usr/bin/env python3
"""Build v2 WebShop scorer data with targeted hard negatives.

This keeps the same state construction and train/valid split as the baseline,
but changes negative sampling to target the observed scorer failure modes:

- navigation/pagination targets confused with item clicks;
- non-buy targets confused with premature `buy now`;
- item-click targets confused with same-page item clicks.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from build_scorer_dataset import action_type, build_states, make_example, sample_negative_actions, split_by_trajectory, write_jsonl


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def add_negative(
    selected: list[dict[str, str]],
    seen: set[str],
    action: str,
    negative_strength: str,
    reason: str,
) -> None:
    if action in seen:
        return
    selected.append(
        {
            "action": action,
            "negative_strength": negative_strength,
            "hard_negative_reason": reason,
        }
    )
    seen.add(action)


def targeted_negatives_for_state(
    state: dict[str, Any],
    max_negatives: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    target = state["target_action"]
    target_type = state["target_action_type"]
    actions = [action for action in state["available_actions"] if action != target]
    by_type: dict[str, list[str]] = {}
    for action in actions:
        by_type.setdefault(action_type(action), []).append(action)

    selected: list[dict[str, str]] = []
    seen: set[str] = set()

    def take_bucket(action_type_name: str, reason: str, limit: int, strength: str = "hard") -> None:
        bucket = list(by_type.get(action_type_name, []))
        rng.shuffle(bucket)
        for action in bucket[:limit]:
            add_negative(selected, seen, action, strength, reason)

    if target_type in {"navigation", "pagination"}:
        take_bucket("item_click", "nav_page_vs_item_click", limit=2)
        take_bucket("buy", "nav_page_vs_premature_buy", limit=1)

    if target_type == "item_click":
        take_bucket("item_click", "same_page_item_click", limit=3)
        take_bucket("buy", "item_click_vs_premature_buy", limit=1)

    if target_type != "buy":
        take_bucket("buy", "premature_buy", limit=1)

    if target_type in {"info", "click_other", "option"}:
        take_bucket("buy", "info_option_vs_premature_buy", limit=1)
        take_bucket("item_click", "info_option_vs_item_click", limit=1)

    if len(selected) < max_negatives:
        fallback = sample_negative_actions(target, state["available_actions"], max_negatives * 2, rng)
        for item in fallback:
            add_negative(
                selected,
                seen,
                item["action"],
                item["negative_strength"],
                "baseline_fallback",
            )
            if len(selected) >= max_negatives:
                break

    return selected[:max_negatives]


def build_examples_v2(
    state_rows: list[dict[str, Any]],
    negatives_per_positive: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    examples = []
    hard_reason_counts = Counter()
    negative_strength_counts = Counter()
    negative_action_type_counts = Counter()

    for state in state_rows:
        examples.append(
            make_example(
                state,
                state["target_action"],
                1,
                "pos",
                "demonstrated_preferred",
            )
        )

        negatives = targeted_negatives_for_state(state, negatives_per_positive, rng)
        for neg_id, negative in enumerate(negatives):
            reason = negative["hard_negative_reason"]
            strength = negative["negative_strength"]
            neg_type = action_type(negative["action"])
            hard_reason_counts[reason] += 1
            negative_strength_counts[strength] += 1
            negative_action_type_counts[neg_type] += 1
            row = make_example(
                state,
                negative["action"],
                0,
                f"neg_{neg_id}",
                "targeted_contrastive_alternative",
                strength,
            )
            row["hard_negative_reason"] = reason
            examples.append(row)

    stats = {
        "num_examples": len(examples),
        "num_positive": sum(1 for ex in examples if int(ex["label"]) == 1),
        "num_negative": sum(1 for ex in examples if int(ex["label"]) == 0),
        "negative_strength_counts": dict(negative_strength_counts.most_common()),
        "negative_action_type_counts": dict(negative_action_type_counts.most_common()),
        "hard_negative_reason_counts": dict(hard_reason_counts.most_common()),
        "negatives_per_positive": negatives_per_positive,
    }
    return examples, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/webshop_demos/il_trajs_finalized_images/il_trajs_finalized_images.jsonl",
    )
    parser.add_argument("--train-states", default="data/processed/scorer_baseline/train_states.jsonl")
    parser.add_argument("--valid-states", default="data/processed/scorer_baseline/valid_states.jsonl")
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Rebuild states from --input instead of reusing baseline train/valid state files.",
    )
    parser.add_argument("--out-dir", default="data/processed/scorer_hardneg_v2")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-history-steps", type=int, default=4)
    parser.add_argument("--negatives-per-positive", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_raw:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Demo file not found: {input_path}")
        states, state_stats = build_states(args)
        train_states, valid_states = split_by_trajectory(states, args.valid_ratio, args.seed)
        state_source = str(input_path)
    else:
        train_states_path = Path(args.train_states)
        valid_states_path = Path(args.valid_states)
        if not train_states_path.exists():
            raise FileNotFoundError(f"Train states file not found: {train_states_path}")
        if not valid_states_path.exists():
            raise FileNotFoundError(f"Valid states file not found: {valid_states_path}")
        train_states = load_jsonl(train_states_path)
        valid_states = load_jsonl(valid_states_path)
        state_source = f"{train_states_path}; {valid_states_path}"
        state_stats = {
            "input": state_source,
            "kept_states": len(train_states) + len(valid_states),
            "skipped": {},
            "target_action_type_counts": dict(
                Counter(str(row.get("target_action_type") or "unknown") for row in train_states + valid_states).most_common()
            ),
            "max_history_steps": "reused_from_baseline_states",
        }

    train_examples, train_example_stats = build_examples_v2(
        train_states,
        args.negatives_per_positive,
        args.seed,
    )
    valid_examples, valid_example_stats = build_examples_v2(
        valid_states,
        args.negatives_per_positive,
        args.seed + 1,
    )

    write_jsonl(out_dir / "train_states.jsonl", train_states)
    write_jsonl(out_dir / "valid_states.jsonl", valid_states)
    write_jsonl(out_dir / "train.jsonl", train_examples)
    write_jsonl(out_dir / "valid.jsonl", valid_examples)

    train_trajs = {int(row["trajectory_id"]) for row in train_states}
    valid_trajs = {int(row["trajectory_id"]) for row in valid_states}
    stats: dict[str, Any] = {
        **state_stats,
        "out_dir": str(out_dir),
        "state_source": state_source,
        "from_raw": args.from_raw,
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "num_train_states": len(train_states),
        "num_valid_states": len(valid_states),
        "train_trajectory_count": len(train_trajs),
        "valid_trajectory_count": len(valid_trajs),
        "trajectory_overlap_count": len(train_trajs & valid_trajs),
        "train_examples": train_example_stats,
        "valid_examples": valid_example_stats,
        "v2_strategy": (
            "targeted hard negatives for nav/page-vs-item, same-page item clicks, "
            "and premature buy; same states/split as baseline"
        ),
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote: {out_dir / 'train_states.jsonl'}")
    print(f"wrote: {out_dir / 'valid_states.jsonl'}")
    print(f"wrote: {out_dir / 'train.jsonl'}")
    print(f"wrote: {out_dir / 'valid.jsonl'}")
    print(f"wrote: {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
