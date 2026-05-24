#!/usr/bin/env python3
"""LoRA SFT for WebShop Yes/No action scorer."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


IGNORE_INDEX = -100


def load_jsonl(path: Path, max_samples: int | None = None, seed: int = 42) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if max_samples is not None and len(rows) > max_samples:
        rng = random.Random(seed)
        rows = rng.sample(rows, max_samples)
    return rows


def encode_sample(
    row: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, list[int]]:
    messages = row["messages"]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    appended_eos = False
    if tokenizer.eos_token_id is not None and (not full_ids or full_ids[-1] != tokenizer.eos_token_id):
        full_ids.append(tokenizer.eos_token_id)
        appended_eos = True

    prefix_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    answer_start = len(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
    answer_start = min(answer_start, len(full_ids))

    labels = [IGNORE_INDEX] * len(full_ids)
    labels[answer_start:] = full_ids[answer_start:]
    if appended_eos:
        labels[-1] = full_ids[-1]

    if len(full_ids) > max_seq_length:
        overflow = len(full_ids) - max_seq_length
        full_ids = full_ids[overflow:]
        labels = labels[overflow:]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class YesNoDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_seq_length: int,
        max_rendered_chars: int,
    ) -> None:
        self.features = []
        self.skipped_overlong_rendered = 0
        self.skipped_no_labels = 0
        for row in rows:
            rendered = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            if max_rendered_chars > 0 and len(rendered) > max_rendered_chars:
                self.skipped_overlong_rendered += 1
                continue
            feature = encode_sample(row, tokenizer, max_seq_length)
            if not any(label != IGNORE_INDEX for label in feature["labels"]):
                self.skipped_no_labels += 1
                continue
            self.features.append(feature)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.features[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", default="data/processed/yesno_scorer_sft/train.jsonl")
    parser.add_argument("--valid-file", default="data/processed/yesno_scorer_sft/valid.jsonl")
    parser.add_argument("--output-dir", default="outputs/yesno_scorer/qwen25_1p5b_lora")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-rendered-chars", type=int, default=12000)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=2000)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=999999)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "torch, transformers, and peft are required for this script. "
            "Run it inside the server training environment."
        ) from exc

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = load_jsonl(Path(args.train_file), args.max_train_samples, args.seed)
    valid_rows = load_jsonl(Path(args.valid_file), args.max_valid_samples, args.seed)

    train_dataset = YesNoDataset(train_rows, tokenizer, args.max_seq_length, args.max_rendered_chars)
    valid_dataset = YesNoDataset(valid_rows, tokenizer, args.max_seq_length, args.max_rendered_chars)
    if len(train_dataset) == 0:
        raise ValueError("No train samples left after tokenization.")
    if len(valid_dataset) == 0:
        raise ValueError("No valid samples left after tokenization.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name_or_path": args.model_name_or_path,
        "train_file": args.train_file,
        "valid_file": args.valid_file,
        "num_train_rows": len(train_rows),
        "num_valid_rows": len(valid_rows),
        "num_train_tokenized": len(train_dataset),
        "num_valid_tokenized": len(valid_dataset),
        "train_skipped_overlong_rendered": train_dataset.skipped_overlong_rendered,
        "valid_skipped_overlong_rendered": valid_dataset.skipped_overlong_rendered,
        "max_seq_length": args.max_seq_length,
        "max_rendered_chars": args.max_rendered_chars,
        "seed": args.seed,
    }
    with (output_dir / "yesno_scorer_train_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    steps_per_epoch = math.ceil(
        len(train_dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    print(f"train_rows: {len(train_rows)} tokenized: {len(train_dataset)}")
    print(f"valid_rows: {len(valid_rows)} tokenized: {len(valid_dataset)}")
    print(f"train_skipped_overlong_rendered: {train_dataset.skipped_overlong_rendered}")
    print(f"valid_skipped_overlong_rendered: {valid_dataset.skipped_overlong_rendered}")
    print(f"approx_steps_per_epoch: {steps_per_epoch}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))

    summary = {
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "final_adapter": str(output_dir / "final_adapter"),
    }
    with (output_dir / "yesno_scorer_train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved final adapter to: {output_dir / 'final_adapter'}")


if __name__ == "__main__":
    main()
