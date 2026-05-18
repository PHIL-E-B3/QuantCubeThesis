"""
Optuna Hyperparameter Search (Generative QLoRA)
================================================
Searches over LoRA hyperparameters to minimise eval loss on the
generative FOMC annotation task.
"""

import os
import gc
import torch
import optuna
from typing import Optional
from functools import partial

from datasets import DatasetDict
from transformers import Trainer, EarlyStoppingCallback, TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from src.training.qlora_trainer import (
    load_model_and_tokenizer,
    get_lora_config,
    get_training_args,
    GenerativeDataCollator,
)


class OptunaPruningCallback(TrainerCallback):
    """Reports epoch-level eval loss to Optuna and raises TrialPruned when warranted."""

    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, metrics=None, **kwargs):
        epoch = int(state.epoch or 0)
        loss = (metrics or {}).get("eval_loss", float("inf"))
        self.trial.report(loss, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {epoch} (loss={loss:.4f})")


def objective(
    trial: optuna.Trial,
    model_name: str,
    dataset: DatasetDict,
    output_base_dir: str,
    search_space: Optional[dict] = None,
) -> float:
    """
    Single Optuna trial: sample hyperparameters, train, return eval loss.
    Lower is better — study direction is 'minimize'.
    """
    if search_space is None:
        search_space = {}

    r = trial.suggest_categorical("lora_r", search_space.get("lora_r", [4, 8, 16]))
    alpha_mult = trial.suggest_categorical(
        "lora_alpha_multiplier", search_space.get("lora_alpha_multiplier", [1, 2])
    )
    lora_dropout = trial.suggest_categorical(
        "lora_dropout", search_space.get("lora_dropout", [0.05, 0.1])
    )
    lr_cfg = search_space.get("learning_rate", {"low": 1e-4, "high": 5e-4, "log": True})
    learning_rate = trial.suggest_float(
        "learning_rate", lr_cfg["low"], lr_cfg["high"], log=lr_cfg.get("log", True)
    )
    weight_decay = trial.suggest_categorical(
        "weight_decay", search_space.get("weight_decay", [0.01, 0.05])
    )
    batch_size = trial.suggest_categorical(
        "batch_size", search_space.get("per_device_train_batch_size", [4, 8])
    )

    lora_config = get_lora_config(r=r, lora_alpha=r * alpha_mult, lora_dropout=lora_dropout)
    model, tokenizer = load_model_and_tokenizer(model_name=model_name, lora_config=lora_config)

    trial_dir = os.path.join(output_base_dir, f"trial_{trial.number}")
    training_args = get_training_args(
        output_dir=trial_dir,
        num_epochs=3,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    data_collator = GenerativeDataCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            OptunaPruningCallback(trial),
        ],
    )

    try:
        trainer.train()
    except optuna.TrialPruned:
        raise

    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss", float("inf"))

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return eval_loss  # minimise


def run_optuna_search(
    model_name: str,
    dataset: DatasetDict,
    output_dir: str = "models/optuna",
    n_trials: int = 25,
    search_space: Optional[dict] = None,
    study_name: str = "fomc_qlora",
) -> optuna.Study:
    """
    Run full Optuna hyperparameter search.

    Args:
        model_name: HuggingFace model identifier.
        dataset: Pre-built DatasetDict with train/validation/test splits.
        output_dir: Where to save trial outputs and the SQLite study DB.
        n_trials: Number of Optuna trials.
        search_space: Custom search ranges (from config['optuna']['search_space']).
        study_name: Name for the Optuna study (used for SQLite key).

    Returns:
        Completed Optuna study with best parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    storage_url = f"sqlite:///{os.path.join(output_dir, f'{study_name}.db')}"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage=storage_url,
        load_if_exists=True,
    )

    obj_fn = partial(
        objective,
        model_name=model_name,
        dataset=dataset,
        output_base_dir=output_dir,
        search_space=search_space,
    )

    study.optimize(obj_fn, n_trials=n_trials)

    print("\n" + "=" * 60)
    print("OPTUNA SEARCH COMPLETE")
    print("=" * 60)
    print(f"Best trial:  #{study.best_trial.number}")
    print(f"Best loss:   {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study
