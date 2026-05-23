#!/usr/bin/env python3
"""Build WebShop step-level action scorer data from human demonstrations.

The first version intentionally covers only discrete admissible-action ranking:
the demonstrated target action must already appear in the state's
available_actions list. Free-form search generation is skipped and counted.

Output files:

- train.jsonl / valid.jsonl: classification examples
- train_states.jsonl / valid_states.jsonl: state-level rows for ranking eval
- stats.json: construction statistics
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTION_RE = re.compile(r"^\s*([a-zA-Z_ ]+)\[(.*)\]\s*$")
AMBIGUOUS_ACTION_TYPES = {"info", "navigation", "pagination"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_instruction(state: str) -> str:
    lines = state.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("instruction"):
            pieces = []
            for nxt in lines[idx + 1 :]:
                stripped = nxt.strip()
                if not stripped:
                    continue
                if (
                    stripped.startswith("[")
                    or stripped.startswith("Page ")
                    or stripped.startswith("Description:")
                    or stripped.startswith("Features:")
                    or stripped.startswith("Reviews:")
                ):
                    break
                pieces.append(stripped)
            return " ".join(pieces).strip()
    return ""


def dedupe_actions(actions: Any) -> list[str]:
    if not isinstance(actions, list):
        return []

    deduped = []
    seen = set()
    for action in actions:
        text = str(action).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def action_inner(action: str) -> str:
    match = ACTION_RE.match(action)
    if not match:
        return ""
    return match.group(2).strip().lower()


def action_type(action: str) -> str:
    text = action.strip().lower()
    inner = action_inner(text)

    if text.startswith("search["):
        return "search"
    if not text.startswith("click["):
        return "other"
    if inner == "buy now":
        return "buy"
    if inner in {"next >", "< prev", "prev", "previous", "next"}:
        return "pagination"
    if inner in {"back to search", "back"}:
        return "navigation"
    if inner in {"description", "features", "reviews"}:
        return "info"
    if inner.startswith("item - ") or re.match(r"^b[0-9a-z]{9}$", inner):
        return "item_click"
    if any(token in inner for token in ("size", "color", "option")):
        return "option"
    return "click_other"


def negative_strength(target: str, candidate: str) -> str:
    target_type = action_type(target)
    candidate_type = action_type(candidate)
    if target_type == candidate_type:
        if target_type in AMBIGUOUS_ACTION_TYPES:
            return "weak"
        return "hard"
    if {target_type, candidate_type} <= AMBIGUOUS_ACTION_TYPES:
        return "weak"
    return "random"


def make_history(
    states: list[Any],
    actions: list[Any],
    step_id: int,
    max_history_steps: int,
) -> list[dict[str, str]]:
    start = max(0, step_id - max_history_steps)
    history = []
    for hist_id in range(start, step_id):
        history.append(
            {
                "step_id": hist_id,
                "observation": str(states[hist_id]).strip(),
                "action": str(actions[hist_id]).strip(),
                "action_type": action_type(str(actions[hist_id]).strip()),
            }
        )
    return history


def split_by_trajectory(
    state_rows: list[dict[str, Any]],
    valid_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_traj: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        by_traj[int(row["trajectory_id"])].append(row)

    traj_ids = list(by_traj)
    rng.shuffle(traj_ids)
    n_valid = max(1, int(len(traj_ids) * valid_ratio)) if traj_ids else 0
    valid_ids = set(traj_ids[:n_valid])

    train = []
    valid = []
    for row in state_rows:
        if int(row["trajectory_id"]) in valid_ids:
            valid.append(row)
        else:
            train.append(row)
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def sample_negative_actions(
    target_action: str,
    available_actions: list[str],
    negatives_per_positive: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    candidates = [action for action in available_actions if action != target_action]
    rng.shuffle(candidates)

    same_type = []
    weak = []
    random_negatives = []
    target_type = action_type(target_action)
    for action in candidates:
        strength = negative_strength(target_action, action)
        if action_type(action) == target_type and strength == "hard":
            same_type.append(action)
        elif strength == "weak":
            weak.append(action)
        else:
            random_negatives.append(action)

    selected = []
    seen = set()
    for bucket in (same_type, weak, random_negatives):
        for action in bucket:
            if action in seen:
                continue
            selected.append(
                {
                    "action": action,
                    "negative_strength": negative_strength(target_action, action),
                }
            )
            seen.add(action)
            if len(selected) >= negatives_per_positive:
                return selected
    return selected


def make_example(
    state: dict[str, Any],
    candidate_action: str,
    label: int,
    sample_id_suffix: str,
    preference_type: str,
    negative_strength_value: str | None = None,
) -> dict[str, Any]:
    row = {
        "sample_id": f"{state['state_id']}_{sample_id_suffix}",
        "state_id": state["state_id"],
        "trajectory_id": state["trajectory_id"],
        "step_id": state["step_id"],
        "instruction": state["instruction"],
        "history": state["history"],
        "observation": state["observation"],
        "candidate_action": candidate_action,
        "candidate_action_type": action_type(candidate_action),
        "target_action": state["target_action"],
        "target_action_type": state["target_action_type"],
        "label": label,
        "preference_type": preference_type,
        "source": state["source"],
    }
    if negative_strength_value is not None:
        row["negative_strength"] = negative_strength_value
    return row


def build_states(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = load_jsonl(Path(args.input))
    state_rows = []
    stats = Counter()
    target_type_counts = Counter()
    available_action_count_hist = Counter()

    for traj_id, obj in enumerate(raw_rows):
        states = obj.get("states") or []
        actions = obj.get("actions_translate") or obj.get("actions") or []
        available_actions_by_step = obj.get("available_actions") or []
        n_steps = min(len(states), len(actions))
        if n_steps == 0:
            stats["empty_trajectory"] += 1
            continue

        instruction = extract_instruction(str(states[0]))
        for step_id in range(n_steps):
            target_action = str(actions[step_id]).strip()
            observation = str(states[step_id]).strip()
            if not target_action or not observation:
                stats["missing_target_or_observation"] += 1
                continue

            raw_available = (
                available_actions_by_step[step_id]
                if step_id < len(available_actions_by_step)
                else []
            )
            available_actions = dedupe_actions(raw_available)
            available_action_count_hist[len(available_actions)] += 1
            target_type = action_type(target_action)

            if target_action not in available_actions:
                stats["target_not_in_available_actions"] += 1
                if target_type == "search":
                    stats["free_form_search_skipped"] += 1
                continue

            if len([a for a in available_actions if a != target_action]) == 0:
                stats["no_contrastive_alternative"] += 1
                continue

            state_id = f"traj_{traj_id:05d}_step_{step_id:03d}"
            target_type_counts[target_type] += 1
            state_rows.append(
                {
                    "state_id": state_id,
                    "trajectory_id": traj_id,
                    "step_id": step_id,
                    "instruction": instruction,
                    "history": make_history(states, actions, step_id, args.max_history_steps),
                    "observation": observation,
                    "available_actions": available_actions,
                    "available_action_types": [action_type(action) for action in available_actions],
                    "target_action": target_action,
                    "target_action_type": target_type,
                    "source": "webshop_human_demo_il_trajs_finalized_images",
                }
            )
            stats["kept_states"] += 1

    summary = {
        "input": args.input,
        "raw_trajectories": len(raw_rows),
        "kept_states": stats["kept_states"],
        "skipped": {
            "empty_trajectory": stats["empty_trajectory"],
            "missing_target_or_observation": stats["missing_target_or_observation"],
            "target_not_in_available_actions": stats["target_not_in_available_actions"],
            "free_form_search_skipped": stats["free_form_search_skipped"],
            "no_contrastive_alternative": stats["no_contrastive_alternative"],
        },
        "target_action_type_counts": dict(target_type_counts.most_common()),
        "available_action_count_hist": dict(sorted(available_action_count_hist.items())),
        "max_history_steps": args.max_history_steps,
    }
    return state_rows, summary


def build_examples(
    state_rows: list[dict[str, Any]],
    negatives_per_positive: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    examples = []
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

        negatives = sample_negative_actions(
            state["target_action"],
            state["available_actions"],
            negatives_per_positive,
            rng,
        )
        for neg_id, negative in enumerate(negatives):
            negative_strength_counts[negative["negative_strength"]] += 1
            negative_action_type_counts[action_type(negative["action"])] += 1
            examples.append(
                make_example(
                    state,
                    negative["action"],
                    0,
                    f"neg_{neg_id}",
                    "sampled_contrastive_alternative",
                    negative["negative_strength"],
                )
            )

    stats = {
        "num_examples": len(examples),
        "num_positive": sum(1 for ex in examples if int(ex["label"]) == 1),
        "num_negative": sum(1 for ex in examples if int(ex["label"]) == 0),
        "negative_strength_counts": dict(negative_strength_counts.most_common()),
        "negative_action_type_counts": dict(negative_action_type_counts.most_common()),
        "negatives_per_positive": negatives_per_positive,
    }
    return examples, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/webshop_demos/il_trajs_finalized_images/il_trajs_finalized_images.jsonl",
    )
    parser.add_argument("--out-dir", default="data/processed/scorer_baseline")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-history-steps", type=int, default=4)
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Demo file not found: {input_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    states, state_stats = build_states(args)
    train_states, valid_states = split_by_trajectory(states, args.valid_ratio, args.seed)

    train_examples, train_example_stats = build_examples(
        train_states,
        args.negatives_per_positive,
        args.seed,
    )
    valid_examples, valid_example_stats = build_examples(
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
    stats = {
        **state_stats,
        "out_dir": str(out_dir),
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "num_train_states": len(train_states),
        "num_valid_states": len(valid_states),
        "train_trajectory_count": len(train_trajs),
        "valid_trajectory_count": len(valid_trajs),
        "trajectory_overlap_count": len(train_trajs & valid_trajs),
        "train_examples": train_example_stats,
        "valid_examples": valid_example_stats,
        "scope": (
            "discrete admissible-action preference scoring; free-form search "
            "steps are skipped when target_action is not in available_actions"
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
