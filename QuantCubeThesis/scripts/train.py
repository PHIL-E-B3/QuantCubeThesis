"""
Main Training Script
====================
Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --task sen
    python scripts/train.py --config configs/default.yaml --task dir --optuna
"""

import argparse
import sys
import os
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import load_labels, create_label_maps, build_classification_dataset
from src.training.qlora_trainer import (
    load_model_and_tokenizer,
    get_lora_config,
    get_training_args,
    train,
    benchmark_baseline,
)
from src.training.hyperparameter_search import run_optuna_search


def main():
    parser = argparse.ArgumentParser(description="FOMC Sentiment QLoRA Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--task", type=str, default="sen",
                        choices=["top", "ten", "sen", "com", "hor", "ris", "wid"],
                        help="Which label field to train on (abbreviated field name)")
    parser.add_argument("--optuna", action="store_true",
                        help="Run Optuna hyperparameter search instead of single train")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline benchmark before training")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to labels CSV (overrides config)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Resolve paths
    labels_path = args.labels or os.path.join(config["paths"]["data_labels"], "labels.json")
    model_name = config["model"]["name"]
    max_length = config["model"]["max_seq_length"]

    print(f"Model:      {model_name}")
    print(f"Task:       {args.task}")
    print(f"Labels:     {labels_path}")
    print(f"Max length: {max_length}")
    print(f"Device:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print()

    # Load labels and create mappings
    df = load_labels(labels_path)
    all_maps = create_label_maps(config)

    if args.task not in all_maps:
        raise ValueError(
            f"Task '{args.task}' has no valid values defined in config labels. "
            f"Fill in configs/default.yaml labels.{args.task} first."
        )

    label_map = all_maps[args.task]
    label_column = args.task
    id2label = {v: k for k, v in label_map.items()}
    num_labels = len(label_map)

    # Load tokenizer for dataset building
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset
    dataset = build_classification_dataset(
        df=df,
        tokenizer=tokenizer,
        label_column=label_column,
        label_map=label_map,
        max_length=max_length,
    )

    # Optional baseline benchmark
    if args.baseline:
        benchmark_baseline(model_name, dataset, num_labels, label_map, id2label)

    if args.optuna:
        # ── Hyperparameter search ───────────────────────────────
        study = run_optuna_search(
            model_name=model_name,
            dataset=dataset,
            num_labels=num_labels,
            label2id=label_map,
            id2label=id2label,
            output_dir=os.path.join(config["paths"]["model_output"], "optuna"),
            n_trials=config["optuna"]["n_trials"],
            search_space=config["optuna"]["search_space"],
        )

        # Retrain with best params
        print("\n=== Retraining with best parameters ===")
        best = study.best_params
        lora_config = get_lora_config(
            r=best["lora_r"],
            lora_alpha=best["lora_r"] * best["lora_alpha_multiplier"],
            lora_dropout=best["lora_dropout"],
        )
        model, tokenizer = load_model_and_tokenizer(
            model_name, num_labels, label_map, id2label, lora_config,
        )
        training_args = get_training_args(
            output_dir=os.path.join(config["paths"]["model_output"], "best"),
            num_epochs=config["training"]["num_epochs"],
            batch_size=best["batch_size"],
            learning_rate=best["learning_rate"],
        )
        train(model, tokenizer, dataset,
              output_dir=os.path.join(config["paths"]["model_output"], "best"),
              training_args=training_args)

    else:
        # ── Single training run with config defaults ────────────
        lora_cfg = config["lora"]
        lora_config = get_lora_config(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
        )
        model, tokenizer = load_model_and_tokenizer(
            model_name, num_labels, label_map, id2label, lora_config,
        )
        output_dir = os.path.join(config["paths"]["model_output"], args.task)
        train(model, tokenizer, dataset, output_dir=output_dir)


if __name__ == "__main__":
    main()
