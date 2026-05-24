#!/usr/bin/env python3
"""Evaluate a LoRA Yes/No scorer by ranking admissible WebShop actions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
    if not history:
        return "None"
    chunks = []
    for item in history:
        step_id = item.get("step_id", "?")
        action = str(item.get("action", ""))
        obs = truncate(item.get("observation", ""), max_history_chars)
        chunks.append(f"Step {step_id} action: {action}\nStep {step_id} observation: {obs}")
    return "\n\n".join(chunks)


def action_type_for_state_action(state: dict[str, Any], action: str) -> str:
    actions = state.get("available_actions") or []
    types = state.get("available_action_types") or []
    for idx, candidate in enumerate(actions):
        if candidate == action and idx < len(types):
            return str(types[idx])
    return "unknown"


def make_prompt(
    state: dict[str, Any],
    action: str,
    action_type: str,
    tokenizer: Any,
    max_observation_chars: int,
    max_history_chars: int,
) -> str:
    user = (
        "You are judging whether a candidate WebShop action is a good next action.\n"
        "Use only the task, recent history, current observation, and candidate action.\n"
        "Answer with exactly one word: Yes or No.\n\n"
        f"Task:\n{state.get('instruction', '')}\n\n"
        f"Recent history:\n{format_history(state.get('history') or [], max_history_chars)}\n\n"
        f"Current observation:\n{truncate(state.get('observation', ''), max_observation_chars)}\n\n"
        f"Candidate action:\n{action}\n\n"
        f"Candidate action type: {action_type}\n\n"
        "Is this candidate action a good next action?"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def yes_no_token_ids(tokenizer: AutoTokenizer) -> tuple[int, int]:
    yes_ids = tokenizer("Yes", add_special_tokens=False)["input_ids"]
    no_ids = tokenizer("No", add_special_tokens=False)["input_ids"]
    if len(yes_ids) != 1 or len(no_ids) != 1:
        yes_ids = tokenizer(" Yes", add_special_tokens=False)["input_ids"]
        no_ids = tokenizer(" No", add_special_tokens=False)["input_ids"]
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(f"Cannot find single-token Yes/No ids: Yes={yes_ids}, No={no_ids}")
    return yes_ids[0], no_ids[0]


def score_prompts(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    yes_id: int,
    no_id: int,
    batch_size: int,
    max_seq_length: int,
) -> list[float]:
    import torch

    scores = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            ).to(device)
            logits = model(**encoded).logits
            lengths = encoded["attention_mask"].sum(dim=1) - 1
            next_logits = logits[torch.arange(logits.shape[0], device=device), lengths]
            pair = next_logits[:, [yes_id, no_id]]
            probs = torch.softmax(pair, dim=-1)[:, 0]
            scores.extend(float(x) for x in probs.detach().cpu())
    return scores


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


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["ranking"]["overall"]
    lines = [
        "# Yes/No LoRA Scorer Ranking",
        "",
        f"States: {summary['num_states']}",
        f"Base model: `{summary['base_model']}`",
        f"Adapter: `{summary['adapter']}`",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| top1 | {overall['top1']:.4f} |",
        f"| top3 | {overall['top3']:.4f} |",
        f"| mrr | {overall['mrr']:.4f} |",
        f"| mean_rank | {overall['mean_rank']:.4f} |",
        f"| mean_num_actions | {overall['mean_num_actions']:.4f} |",
        "",
        "## By Target Action Type",
        "",
        "| Type | states | top1 | top3 | mrr | mean_rank | mean_actions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
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
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--states", default="data/processed/scorer_baseline/valid_states.jsonl")
    parser.add_argument("--out-json", default="data/processed/scorer_baseline/yesno_scorer_ranking.json")
    parser.add_argument("--out-md", default="reports/yesno_scorer_ranking.md")
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-observation-chars", type=int, default=4000)
    parser.add_argument("--max-history-chars", type=int, default=800)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "torch, transformers, and peft are required for this script. "
            "Run it inside the server training environment."
        ) from exc

    states = load_jsonl(Path(args.states), args.max_states)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    yes_id, no_id = yes_no_token_ids(tokenizer)
    ranks = []
    by_type_states: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type_ranks: dict[str, list[int]] = defaultdict(list)
    examples = []

    for state_idx, state in enumerate(states):
        actions = list(state.get("available_actions") or [])
        prompts = [
            make_prompt(
                state,
                action,
                action_type_for_state_action(state, action),
                tokenizer,
                args.max_observation_chars,
                args.max_history_chars,
            )
            for action in actions
        ]
        scores = score_prompts(
            model,
            tokenizer,
            prompts,
            yes_id,
            no_id,
            args.batch_size,
            args.max_seq_length,
        )
        scored = sorted(zip(actions, scores), key=lambda item: (-item[1], item[0]))
        rank = next(idx for idx, (action, _) in enumerate(scored, start=1) if action == state["target_action"])
        ranks.append(rank)
        target_type = str(state.get("target_action_type") or "unknown")
        by_type_states[target_type].append(state)
        by_type_ranks[target_type].append(rank)
        if state_idx < 20:
            examples.append(
                {
                    "state_id": state.get("state_id"),
                    "target_action": state.get("target_action"),
                    "target_action_type": target_type,
                    "rank": rank,
                    "top5": [{"action": action, "score": score} for action, score in scored[:5]],
                }
            )

    summary = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "states_file": args.states,
        "num_states": len(states),
        "yes_token_id": yes_id,
        "no_token_id": no_id,
        "ranking": {
            "overall": summarize_ranks(states, ranks),
            "by_target_action_type": {
                key: summarize_ranks(by_type_states[key], by_type_ranks[key])
                for key in sorted(by_type_states)
            },
        },
        "examples": examples,
    }
    write_json(Path(args.out_json), summary)
    write_markdown(Path(args.out_md), summary)
    print(json.dumps(summary["ranking"], indent=2, ensure_ascii=False))
    print(f"wrote: {args.out_json}")
    print(f"wrote: {args.out_md}")


if __name__ == "__main__":
    main()
