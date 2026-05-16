"""
SFT Fine-Tuning Script
=======================
Fine-tunes Llama 3.1 8B with QLoRA to generate 7-field JSON
classifications for FOMC sentences, using the P3 prompt template.

Usage:
    python scripts/sft_train.py
    python scripts/sft_train.py --epochs 5 --lr 2e-4
    python scripts/sft_train.py --model unsloth/Meta-Llama-3.1-8B-Instruct
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
# Using transformers.Trainer directly for compatibility across TRL versions

# ── PATHS ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
EVAL_PATH = PROJECT_ROOT / "data" / "eval_labelled_merged.json"

# Prompt name → output directory mapping
PROMPT_CONFIGS = {
    "P3_medium_5shot": PROJECT_ROOT / "models" / "sft_p3",
    "P7_high_5shot":   PROJECT_ROOT / "models" / "sft_p7",
}

# Few-shot example IDs to exclude from training/test
EXAMPLE_IDS = {
    "e5418507-7ab1-476e-8db9-fc28796c584f",
    "99122de6-aab0-49fc-9183-d4bd3fc33e27",
    "c7346ec3-ca84-449c-a104-7e0c8d3543ba",
    "cd96b673-cdc4-4ab3-a428-3880ba0bd1dd",
    "995fddd0-966b-4ea0-9eaf-879c5f7fbeed",
}

# Label fields in the order the model should output them
LABEL_FIELDS = ["top", "ten", "sen", "hor", "com", "ris", "wid"]


def load_prompt_template(prompt_path) -> str:
    """Load prompt template from file."""
    with open(prompt_path, encoding="utf-8") as f:
        return f.read().strip()


def format_label_json(example: dict) -> str:
    """Convert a labelled example's fields into the target JSON string."""
    label = {}
    for field in LABEL_FIELDS:
        val = example[field]
        # Ensure consistent types
        if field == "top" and isinstance(val, str):
            val = [val]
        if field == "hor" and isinstance(val, str):
            val = val.lower() == "true"
        label[field] = val
    return json.dumps(label)


def build_dataset(data: list, prompt_template: str, tokenizer) -> Dataset:
    """Format labelled examples as chat-style training sequences.

    Each training example becomes:
        User: [P3 prompt with sentence]
        Assistant: [correct JSON]
    """
    formatted = []
    for ex in data:
        # Fill in the sentence
        user_content = prompt_template.replace("{sentence}", ex["sentence"])
        target_json = format_label_json(ex)

        # Format as chat messages
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target_json},
        ]

        # Apply chat template to get the full training text
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        formatted.append({
            "id": ex["id"],
            "text": text,
        })

    return Dataset.from_list(formatted)


def main():
    parser = argparse.ArgumentParser(description="SFT Fine-Tuning for FOMC Classification")
    parser.add_argument("--prompt", type=str, default="P3_medium_5shot",
                        choices=list(PROMPT_CONFIGS.keys()),
                        help="Prompt template to use for training")
    parser.add_argument("--model", type=str,
                        default="unsloth/Meta-Llama-3.1-8B-Instruct",
                        help="Base model name")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--max-seq-length", type=int, default=1280,
                        help="Max sequence length for training")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split")
    args = parser.parse_args()

    # Resolve prompt and output dir
    prompt_name = args.prompt
    PROMPT_PATH = PROMPTS_DIR / f"{prompt_name}.txt"
    OUTPUT_DIR = PROMPT_CONFIGS[prompt_name]

    print("=" * 60)
    print("SFT FINE-TUNING: FOMC Sentence Classification")
    print("=" * 60)
    print(f"Model:          {args.model}")
    print(f"Prompt:         {prompt_name}")
    print(f"Epochs:         {args.epochs}")
    print(f"Learning rate:  {args.lr}")
    print(f"Batch size:     {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"Max seq length: {args.max_seq_length}")
    print(f"LoRA rank:      {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"Seed:           {args.seed}")
    print(f"Output:         {OUTPUT_DIR}")
    print()

    # ── 1. LOAD AND SPLIT DATA ──────────────────────────────────────────────
    with open(EVAL_PATH, encoding="utf-8") as f:
        all_data = json.load(f)

    # Exclude few-shot examples
    data = [s for s in all_data if s["id"] not in EXAMPLE_IDS]
    print(f"Loaded {len(all_data)} sentences, {len(data)} after excluding few-shot examples")

    # Stratified 80/20 split by sen
    # Convert sen to string for stratification (handles mixed int/str types)
    sen_labels = [str(s.get("sen", "0")) for s in data]

    train_data, test_data, _, _ = train_test_split(
        data, sen_labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=sen_labels,
    )
    print(f"Train: {len(train_data)} sentences")
    print(f"Test:  {len(test_data)} sentences (held out)")

    # Print label distribution
    from collections import Counter
    train_sen = Counter(str(s.get("sen")) for s in train_data)
    test_sen = Counter(str(s.get("sen")) for s in test_data)
    print(f"\nSEN distribution:")
    print(f"  Train: {dict(sorted(train_sen.items()))}")
    print(f"  Test:  {dict(sorted(test_sen.items()))}")

    # ── 2. LOAD TOKENIZER AND FORMAT DATA ────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # Required for SFT

    prompt_template = load_prompt_template(PROMPT_PATH)
    train_dataset = build_dataset(train_data, prompt_template, tokenizer)
    print(f"Formatted {len(train_dataset)} training examples")

    # Check token lengths
    sample_tokens = tokenizer(train_dataset[0]["text"], return_tensors="pt")
    sample_len = sample_tokens["input_ids"].shape[1]
    print(f"Sample training sequence length: {sample_len} tokens")

    # Check how many exceed max_seq_length
    lengths = []
    for ex in train_dataset:
        toks = tokenizer(ex["text"], return_tensors="pt")
        lengths.append(toks["input_ids"].shape[1])
    lengths = np.array(lengths)
    print(f"Token length stats: min={lengths.min()}, median={int(np.median(lengths))}, "
          f"max={lengths.max()}, >max_seq_length={sum(lengths > args.max_seq_length)}")

    if sum(lengths > args.max_seq_length) > 0:
        print(f"WARNING: {sum(lengths > args.max_seq_length)} sequences exceed max_seq_length={args.max_seq_length}")
        print(f"  Consider increasing --max-seq-length to {int(np.percentile(lengths, 99)) + 50}")

    # ── 3. LOAD MODEL ────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Prepare for QLoRA training
    model = prepare_model_for_kbit_training(model)

    # ── 4. ATTACH LoRA ADAPTERS ──────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ── 5. TRAIN ─────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Truncate training sequences to max_seq_length via tokenizer
    def tokenize_and_truncate(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    train_dataset_tok = train_dataset.map(
        tokenize_and_truncate,
        remove_columns=["text", "id"],
        desc="Tokenizing",
    )

    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tok,
        data_collator=data_collator,
    )

    print(f"\n{'=' * 60}")
    print("STARTING TRAINING")
    print(f"{'=' * 60}")
    print(f"Total steps: {len(train_dataset) // (args.batch_size * args.grad_accum) * args.epochs}")
    print()

    trainer.train()

    # ── 6. SAVE ──────────────────────────────────────────────────────────────
    # Save adapter
    adapter_path = OUTPUT_DIR / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nAdapter saved to {adapter_path}")

    # Save train/test split IDs for reproducibility
    split_info = {
        "seed": args.seed,
        "train_ids": [s["id"] for s in train_data],
        "test_ids": [s["id"] for s in test_data],
        "train_size": len(train_data),
        "test_size": len(test_data),
        "prompt": prompt_name,
    }
    with open(OUTPUT_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Save training args for thesis reproducibility
    train_config = {
        "model": args.model,
        "prompt": prompt_name,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "quantization": "4bit_nf4_double_quant",
        "optimizer": "paged_adamw_8bit",
        "scheduler": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "seed": args.seed,
        "token_length_stats": {
            "min": int(lengths.min()),
            "median": int(np.median(lengths)),
            "max": int(lengths.max()),
            "exceeding_max": int(sum(lengths > args.max_seq_length)),
        },
        "train_size": len(train_data),
        "test_size": len(test_data),
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    with open(OUTPUT_DIR / "training_config.json", "w") as f:
        json.dump(train_config, f, indent=2)

    print(f"\nSplit info saved to {OUTPUT_DIR / 'split_info.json'}")
    print(f"Training config saved to {OUTPUT_DIR / 'training_config.json'}")

    # ── 7. SUMMARY ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Adapter:        {adapter_path}")
    print(f"Train/Test IDs: {OUTPUT_DIR / 'split_info.json'}")
    print(f"Config:         {OUTPUT_DIR / 'training_config.json'}")
    print(f"\nNext step: run evaluation on the {len(test_data)} held-out test sentences:")
    print(f"  python scripts/sft_eval.py")


if __name__ == "__main__":
    main()
