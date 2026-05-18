"""
QLoRA Fine-Tuning Trainer (Generative)
=======================================
Fine-tunes Llama 3.1 8B with 4-bit QLoRA to generate 7-field JSON annotations.
The model learns: sentence + prompt → {"topic": [...], "sentiment": "...", ...}
"""

import os
import gc
import torch
import numpy as np
from typing import Dict, Optional, List

from transformers import (
    AutoModelForCausalLM,
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


# ── Data collator ────────────────────────────────────────────────────────────

class GenerativeDataCollator:
    """
    Pads input_ids, attention_mask, and labels (which already contain -100
    for prompt tokens) to the batch maximum length.
    """

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[dict]) -> dict:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        max_len = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        pad = self.pad_token_id
        padded_ids  = [ids  + [pad]  * (max_len - len(ids))  for ids  in input_ids]
        padded_mask = [mask + [0]    * (max_len - len(mask)) for mask in attention_mask]
        padded_lbl  = [lbl  + [-100] * (max_len - len(lbl))  for lbl  in labels]

        return {
            "input_ids":      torch.tensor(padded_ids,  dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "labels":         torch.tensor(padded_lbl,  dtype=torch.long),
        }


# ── Model helpers ─────────────────────────────────────────────────────────────

def get_quantization_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
) -> LoraConfig:
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


def load_model_and_tokenizer(
    model_name: str,
    lora_config: LoraConfig,
    quantization_config: Optional[BitsAndBytesConfig] = None,
):
    """Load 4-bit quantized CausalLM with LoRA adapters."""
    if quantization_config is None:
        quantization_config = get_quantization_config()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model, tokenizer


def get_training_args(
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    **kwargs,
) -> TrainingArguments:
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
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
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
    if training_args is None:
        training_args = get_training_args(output_dir, **kwargs)

    data_collator = GenerativeDataCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )

    print(f"\nTrain: {len(dataset['train'])}  "
          f"Val: {len(dataset['validation'])}  "
          f"Test: {len(dataset['test'])}")
    trainer.train()

    test_results = trainer.evaluate(dataset["test"])
    print("\nTest results:")
    for k, v in test_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    adapter_path = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nAdapter saved to {adapter_path}")

    return trainer
