"""
QLoRA Fine-Tuning Trainer
=========================
Fine-tunes a quantized LLM for FOMC sentiment classification
using QLoRA (4-bit NF4 quantization + LoRA adapters).
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import DatasetDict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


def get_quantization_config() -> BitsAndBytesConfig:
    """Create 4-bit NF4 quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Saves ~0.4GB via double quantization
    )


def get_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
) -> LoraConfig:
    """Create LoRA configuration.

    Args:
        r: Rank of the low-rank matrices.
        lora_alpha: Scaling factor (controls update magnitude).
        lora_dropout: Dropout rate for regularization.
        target_modules: Which layers to inject LoRA into.
            None defaults to all linear layers (per Dettmers et al.).
    """
    if target_modules is None:
        # All linear layers — shown to outperform attention-only in QLoRA paper
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,
    )


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    lora_config: LoraConfig,
    quantization_config: Optional[BitsAndBytesConfig] = None,
):
    """
    Load a quantized model with LoRA adapters for classification.

    Returns:
        (model, tokenizer) tuple ready for training.
    """
    if quantization_config is None:
        quantization_config = get_quantization_config()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load quantized model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Ensure the classification head uses the pad token for pooling
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare for k-bit training (freeze base, enable gradient checkpointing)
    model = prepare_model_for_kbit_training(model)

    # Inject LoRA adapters
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total
    print(f"\nTrainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")

    return model, tokenizer


def compute_metrics(eval_pred) -> dict:
    """Compute classification metrics for the Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def get_training_args(
    output_dir: str,
    num_epochs: int = 5,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    **kwargs,
) -> TrainingArguments:
    """Create training arguments optimized for RTX 4070 Ti Super (16GB)."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer for QLoRA
        report_to="none",         # Change to "wandb" if using W&B
        remove_unused_columns=False,
        **kwargs,
    )


def train(
    model,
    tokenizer,
    dataset: DatasetDict,
    output_dir: str,
    training_args: Optional[TrainingArguments] = None,
    **kwargs,
) -> Trainer:
    """
    Run the full training loop.

    Args:
        model: PEFT model with LoRA adapters.
        tokenizer: Tokenizer.
        dataset: DatasetDict with train/validation/test splits.
        output_dir: Where to save checkpoints.
        training_args: Optional custom TrainingArguments.

    Returns:
        Trained Trainer instance.
    """
    if training_args is None:
        training_args = get_training_args(output_dir, **kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    print("\n=== Starting QLoRA Fine-Tuning ===")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Val samples:   {len(dataset['validation'])}")
    print(f"Test samples:  {len(dataset['test'])}")
    print(f"Epochs:        {training_args.num_train_epochs}")
    print(f"Batch size:    {training_args.per_device_train_batch_size} "
          f"(effective: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps})")
    print()

    trainer.train()

    # Evaluate on test set
    print("\n=== Test Set Evaluation ===")
    test_results = trainer.evaluate(dataset["test"])
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}")

    # Save the LoRA adapter (not the full model)
    adapter_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nAdapter saved to {adapter_path}")

    return trainer


def benchmark_baseline(
    model_name: str,
    dataset: DatasetDict,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> dict:
    """
    Benchmark the base model BEFORE fine-tuning (zero-shot).
    This gives us a comparison point as recommended in the doc.
    """
    print("\n=== Baseline Benchmark (Pre-Fine-Tune) ===")

    bnb_config = get_quantization_config()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate(dataset["test"])
    print("Baseline results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # Free memory
    del model
    torch.cuda.empty_cache()

    return results
