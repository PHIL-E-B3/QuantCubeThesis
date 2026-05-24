"""
Main Training Script (Generative QLoRA)
========================================
Fine-tunes Llama 3.1 8B with 4-bit QLoRA to generate 7-field JSON annotations.
Training data: initial seed sentences (data/QuantCube_Seed_Labelled/).
Model learns: sentence + prompt → {"topic": [...], "sentiment": "...", ...}

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --prompt prompts/P7_high_5shot_final.txt
    python scripts/train.py --config configs/default.yaml --prompt prompts/P3_medium_5shot_final.txt --optuna
    python scripts/train.py --config configs/default.yaml --baseline
"""

import argparse
import sys
import os
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import load_labels, build_generative_dataset
from src.training.qlora_trainer import (
    load_model_and_tokenizer,
    get_lora_config,
    get_training_args,
    train,
    GenerativeDataCollator,
)
from src.training.hyperparameter_search import run_optuna_search


def main():
    parser = argparse.ArgumentParser(description="FOMC Generative QLoRA Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--prompt", type=str,
        default="prompts/P5_v27.txt",
        help="Prompt template file with {sentence} placeholder",
    )
    parser.add_argument("--optuna", action="store_true",
                        help="Run Optuna hyperparameter search")
    parser.add_argument("--baseline", action="store_true",
                        help="Print dataset stats and exit (no training)")
    parser.add_argument("--retrain-best", action="store_true",
                        help="Skip Optuna search; retrain with best params already in DB")
    parser.add_argument("--warm-start", action="store_true",
                        help="Run new Optuna study (fomc_qlora_v2) seeded with best params from fomc_qlora")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name  = config["model"]["name"]
    max_length  = config["model"]["max_seq_length"]
    data_paths  = config["paths"]["seed_data_merged"]
    prompt_path = args.prompt

    # seed_data_merged: training-only seed data (never includes the eval set)
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    print(f"Model:      {model_name}")
    print(f"Prompt:     {prompt_path}")
    print(f"Data:       {data_paths}")
    print(f"Max length: {max_length}")
    print(f"Device:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print()

    # ── Load data (combine + deduplicate across all seed files) ───────────────
    import json, pandas as pd
    seen_ids, all_records = set(), []
    for p in data_paths:
        with open(p, encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            if r["id"] not in seen_ids:
                all_records.append(r)
                seen_ids.add(r["id"])
    df = pd.DataFrame(all_records)
    print(f"Loaded {len(df)} labelled sentences from {len(data_paths)} file(s)")
    print(f"SEN distribution:\n{df['sen'].value_counts().to_string()}\n")

    with open(prompt_path, encoding="utf-8") as f:
        prompt_template = f.read().strip()

    # ── Build generative dataset (tokenizer needed first) ─────────────────────
    # We load the tokenizer before building the dataset to apply the chat template.
    # The model itself is loaded later (after dataset is ready) to save GPU memory
    # during the potentially slow tokenization step.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    dataset = build_generative_dataset(
        df=df,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        max_length=max_length,
    )

    if args.baseline:
        print("\nDataset built. Exiting (--baseline mode).")
        return

    # ── Single run or Optuna search ───────────────────────────────────────────
    if args.retrain_best:
        import optuna
        db_path = os.path.join(config["paths"]["model_output"], "optuna", "fomc_qlora.db")
        study = optuna.load_study(
            study_name="fomc_qlora",
            storage=f"sqlite:///{db_path}",
        )
        best = study.best_params
        print(f"\n=== Retraining with best parameters (trial #{study.best_trial.number}, loss={study.best_value:.4f}) ===")
        print(f"  Params: {best}")
        lora_config = get_lora_config(
            r=best["lora_r"],
            lora_alpha=best["lora_r"] * best["lora_alpha_multiplier"],
            lora_dropout=best["lora_dropout"],
        )
        model, _ = load_model_and_tokenizer(model_name, lora_config)
        tr_cfg = config["training"]
        training_args = get_training_args(
            output_dir=os.path.join(config["paths"]["model_output"], "best"),
            num_epochs=tr_cfg["num_epochs"],
            batch_size=best["batch_size"],
            learning_rate=best["learning_rate"],
            weight_decay=best["weight_decay"],
            gradient_accumulation_steps=tr_cfg["gradient_accumulation_steps"],
            warmup_ratio=tr_cfg["warmup_ratio"],
        )
        train(model, tokenizer, dataset,
              output_dir=os.path.join(config["paths"]["model_output"], "best"),
              training_args=training_args)
        return

    if args.warm_start or args.optuna:
        if args.warm_start:
            import sqlite3 as _sqlite3, json as _json
            old_db = os.path.join(config["paths"]["model_output"], "optuna", "fomc_qlora.db")
            conn = _sqlite3.connect(old_db)
            best_row = conn.execute("""
                SELECT t.trial_id FROM trials t
                JOIN trial_values tv ON t.trial_id = tv.trial_id
                WHERE t.state = 'COMPLETE'
                ORDER BY tv.value ASC LIMIT 1
            """).fetchone()
            raw_params = conn.execute(
                "SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=?",
                (best_row[0],)
            ).fetchall()
            conn.close()
            seed_params = {}
            for name, value, dist_json in raw_params:
                dist = _json.loads(dist_json)
                if dist.get("name") == "CategoricalDistribution":
                    seed_params[name] = dist["attributes"]["choices"][int(value)]
                else:
                    seed_params[name] = value
            study_name  = "fomc_qlora_v2"
            enqueue     = seed_params
            print(f"\nWarm-start seed params from fomc_qlora: {seed_params}")
            print(f"  OPTUNA WARM-START SEARCH (study: {study_name})")
        else:
            study_name  = "fomc_qlora"
            enqueue     = None
            print(f"  OPTUNA HYPERPARAMETER SEARCH (study: {study_name})")

        n_trials = config["optuna"]["n_trials"]
        print(f"\n{'='*60}")
        print(f"  Trials: {n_trials}  |  Train: {len(dataset['train'])}  "
              f"Val: {len(dataset['validation'])}")
        print(f"{'='*60}\n")

        study = run_optuna_search(
            model_name=model_name,
            dataset=dataset,
            output_dir=os.path.join(config["paths"]["model_output"], "optuna"),
            n_trials=n_trials,
            search_space=config["optuna"]["search_space"],
            study_name=study_name,
            enqueue_params=enqueue,
        )

        print("\n=== Retraining with best parameters ===")
        best = study.best_params
        lora_config = get_lora_config(
            r=best["lora_r"],
            lora_alpha=best["lora_r"] * best["lora_alpha_multiplier"],
            lora_dropout=best["lora_dropout"],
        )
        model, _ = load_model_and_tokenizer(model_name, lora_config)
        tr_cfg = config["training"]
        training_args = get_training_args(
            output_dir=os.path.join(config["paths"]["model_output"], "best"),
            num_epochs=tr_cfg["num_epochs"],
            batch_size=best["batch_size"],
            learning_rate=best["learning_rate"],
            weight_decay=best["weight_decay"],
            gradient_accumulation_steps=tr_cfg["gradient_accumulation_steps"],
            warmup_ratio=tr_cfg["warmup_ratio"],
        )
        train(model, tokenizer, dataset,
              output_dir=os.path.join(config["paths"]["model_output"], "best"),
              training_args=training_args)

    else:
        lora_cfg = config["lora"]
        lora_config = get_lora_config(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
        )
        model, _ = load_model_and_tokenizer(model_name, lora_config)

        tr_cfg = config["training"]
        output_dir = os.path.join(config["paths"]["model_output"], "sft")
        training_args = get_training_args(
            output_dir=output_dir,
            num_epochs=tr_cfg["num_epochs"],
            batch_size=tr_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=tr_cfg["gradient_accumulation_steps"],
            learning_rate=tr_cfg["learning_rate"],
            weight_decay=tr_cfg["weight_decay"],
            warmup_ratio=tr_cfg["warmup_ratio"],
        )
        train(model, tokenizer, dataset, output_dir=output_dir,
              training_args=training_args)


if __name__ == "__main__":
    main()
