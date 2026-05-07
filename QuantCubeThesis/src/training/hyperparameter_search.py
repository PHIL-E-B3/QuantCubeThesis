"""
Optuna Hyperparameter Search
=============================
Searches over QLoRA hyperparameters (rank, alpha, dropout, lr, batch size)
to find optimal configuration for the FOMC classification task.
"""

import os
import gc
import torch
import optuna
from typing import Dict, Optional
from functools import partial

from datasets import DatasetDict

from src.training.qlora_trainer import (
    load_model_and_tokenizer,
    get_lora_config,
    get_quantization_config,
    get_training_args,
    compute_metrics,
)
from transformers import Trainer, EarlyStoppingCallback


def objective(
    trial: optuna.Trial,
    model_name: str,
    dataset: DatasetDict,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    output_base_dir: str,
    search_space: Optional[dict] = None,
) -> float:
    """
    Single Optuna trial: sample hyperparameters, train, return val F1.
    """
    if search_space is None:
        search_space = {}

    # ── Sample hyperparameters ──────────────────────────────────
    r_choices = search_space.get("lora_r", [4, 8, 16, 32])
    r = trial.suggest_categorical("lora_r", r_choices)

    alpha_mult_choices = search_space.get("lora_alpha_multiplier", [1, 2])
    alpha_mult = trial.suggest_categorical("lora_alpha_multiplier", alpha_mult_choices)
    lora_alpha = r * alpha_mult

    dropout_choices = search_space.get("lora_dropout", [0.0, 0.05, 0.1])
    lora_dropout = trial.suggest_categorical("lora_dropout", dropout_choices)

    lr_config = search_space.get("learning_rate", {"low": 1e-4, "high": 5e-4})
    learning_rate = trial.suggest_float(
        "learning_rate",
        lr_config["low"],
        lr_config["high"],
        log=lr_config.get("log", True),
    )

    bs_choices = search_space.get("per_device_train_batch_size", [4, 8, 16])
    batch_size = trial.suggest_categorical("batch_size", bs_choices)

    # ── Build model with sampled config ─────────────────────────
    lora_config = get_lora_config(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        lora_config=lora_config,
    )

    # ── Training ────────────────────────────────────────────────
    trial_dir = os.path.join(output_base_dir, f"trial_{trial.number}")
    training_args = get_training_args(
        output_dir=trial_dir,
        num_epochs=3,   # Fewer epochs for search (speed)
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # ── Evaluate ────────────────────────────────────────────────
    eval_results = trainer.evaluate()
    f1_macro = eval_results.get("eval_f1_macro", 0.0)

    # ── Cleanup GPU memory ──────────────────────────────────────
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return f1_macro


def run_optuna_search(
    model_name: str,
    dataset: DatasetDict,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    output_dir: str = "models/optuna",
    n_trials: int = 30,
    search_space: Optional[dict] = None,
    study_name: str = "fomc_qlora",
) -> optuna.Study:
    """
    Run full Optuna hyperparameter search.

    Args:
        model_name: HuggingFace model identifier.
        dataset: DatasetDict with train/validation splits.
        num_labels: Number of classification labels.
        label2id/id2label: Label mappings.
        output_dir: Where to save trial outputs.
        n_trials: Number of Optuna trials.
        search_space: Custom search ranges (from config).
        study_name: Name for the Optuna study.

    Returns:
        Completed Optuna study with best parameters.
    """
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximize F1
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    obj_fn = partial(
        objective,
        model_name=model_name,
        dataset=dataset,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        output_base_dir=output_dir,
        search_space=search_space,
    )

    study.optimize(obj_fn, n_trials=n_trials)

    # ── Report ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OPTUNA SEARCH COMPLETE")
    print("=" * 60)
    print(f"Best trial:  #{study.best_trial.number}")
    print(f"Best F1:     {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study
