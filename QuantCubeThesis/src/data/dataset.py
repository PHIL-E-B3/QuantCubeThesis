"""
Dataset Builder
===============
Converts labelled FOMC sentences into HuggingFace Datasets
ready for QLoRA fine-tuning (generative CausalLM approach).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split


def load_labels(labels_path: str) -> pd.DataFrame:
    """
    Load hand-labelled data from a JSON file.

    Required fields per record: id, sentence
    Label fields (abbreviated): top, ten, sen, com, hor, ris, wid
    """
    import json
    with open(labels_path, encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)

    required_cols = ["id", "sentence"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return df


# Mapping from abbreviated field names to human-readable names
LABEL_FIELDS = {
    "top": "topic",
    "ten": "tense",
    "sen": "sentiment",
    "com": "commitment",
    "hor": "horizon",
    "ris": "risk_balance",
    "wid": "width",
}


def create_label_maps(config: dict) -> Dict[str, Dict[str, int]]:
    """
    Build label-to-id mappings for all label dimensions from config.

    All label values are cast to strings so that mixed-type YAML lists
    (e.g. ``[-2, -1, 0, 1, 2, "na"]`` or ``[true, false]``) produce
    consistent ``{str: int}`` mappings.

    Returns:
        Dict mapping each abbreviated field name to its {label_str: int} map.
        e.g. {"sen": {"-2": 0, "-1": 1, ...}, "top": {...}, ...}
    """
    return {
        field: {str(label): i for i, label in enumerate(values)}
        for field, values in config["labels"].items()
        if values  # skip fields with empty value lists
    }


def build_classification_dataset(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    label_column: str,
    label_map: Dict[str, int],
    max_length: int = 256,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[DatasetDict, DataCollatorWithPadding]:
    """
    Build a HuggingFace DatasetDict from labelled DataFrame.

    Args:
        df: DataFrame with 'sentence' and label_column columns.
        tokenizer: Pre-loaded tokenizer.
        label_column: Column name containing the labels.
        label_map: Mapping from label strings to integer IDs.
        max_length: Maximum token sequence length.
        test_size: Fraction for test split.
        val_size: Fraction for validation split (from remaining after test).
        random_state: Random seed.

    Returns:
        (DatasetDict with train/validation/test splits, DataCollatorWithPadding).
        Pass the collator to Trainer so sequences are padded dynamically per batch
        rather than globally to max_length — reduces wasted computation on short sentences.
    """
    # Filter to rows that have a valid label
    # Cast to str so mixed-type columns (int/str/bool) match the str keys in label_map
    df[label_column] = df[label_column].astype(str)
    valid_mask = df[label_column].isin(label_map.keys())
    df_valid = df[valid_mask].copy()
    df_valid["label"] = df_valid[label_column].map(label_map)

    if len(df_valid) == 0:
        raise ValueError(f"No valid labels found in column '{label_column}'")

    print(f"Label distribution for '{label_column}':")
    print(df_valid[label_column].value_counts().to_string())
    print()

    # Stratified splits
    train_df, test_df = train_test_split(
        df_valid, test_size=test_size,
        stratify=df_valid["label"], random_state=random_state,
    )
    adjusted_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=adjusted_val_size,
        stratify=train_df["label"], random_state=random_state,
    )

    print(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    def tokenize_split(split_df: pd.DataFrame) -> Dataset:
        texts = split_df["sentence"].tolist()
        labels = split_df["label"].tolist()

        # No padding here — DataCollatorWithPadding pads each batch to its
        # longest sequence, which is faster than global max_length padding.
        encodings = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
        )

        return Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        })

    dataset = DatasetDict({
        "train": tokenize_split(train_df),
        "validation": tokenize_split(val_df),
        "test": tokenize_split(test_df),
    })
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    return dataset, collator


# Mapping from data file field names (short) to prompt/output field names (full)
FIELD_REMAP = {
    "top": "topic",
    "ten": "tense",
    "sen": "sentiment",
    "hor": "horizon",
    "com": "commitment",
    "ris": "risk",
    "wid": "width",
}


def format_target_json(record: dict) -> str:
    """Build the target JSON string with full field names from a labelled record."""
    out = {}
    for short, long in FIELD_REMAP.items():
        val = record.get(short)
        if short == "top" and isinstance(val, str):
            val = [val]
        if short == "hor" and isinstance(val, str):
            val = val.lower() == "true"
        out[long] = val
    return json.dumps(out)


def build_generative_dataset(
    df: pd.DataFrame,
    tokenizer,
    prompt_template: str,
    max_length: int = 1280,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> DatasetDict:
    """
    Build a DatasetDict for generative (CausalLM) fine-tuning.

    Each example is the full chat sequence tokenized with labels masked to -100
    for the user prompt — loss is computed only on the JSON response tokens.

    Args:
        df: Labelled DataFrame with abbreviated field columns (top, ten, sen, ...).
        tokenizer: Pre-loaded tokenizer with chat template support.
        prompt_template: Full prompt string with a {sentence} placeholder.
        max_length: Maximum sequence length (prompt + JSON response).
        test_size: Fraction held out as test set.
        val_size: Fraction of the remainder used as validation.
        random_state: Reproducibility seed.

    Returns:
        DatasetDict with train / validation / test splits.
        Each example has: input_ids, attention_mask, labels (with -100 on prompt).
    """
    df = df.copy()

    # Filter to rows that have all required label fields filled in
    required = list(FIELD_REMAP.keys())
    valid_mask = df[required].notna().all(axis=1)
    df = df[valid_mask].copy()

    if len(df) == 0:
        raise ValueError("No rows with all required label fields present.")

    # Stratify by sentiment (most imbalanced field)
    df["_strat"] = df["sen"].astype(str)

    train_df, test_df = train_test_split(
        df, test_size=test_size,
        stratify=df["_strat"], random_state=random_state,
    )
    adjusted_val = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=adjusted_val,
        stratify=train_df["_strat"], random_state=random_state,
    )

    print(f"Generative split — train: {len(train_df)}, "
          f"val: {len(val_df)}, test: {len(test_df)}")

    def make_hf_dataset(split_df: pd.DataFrame) -> Dataset:
        all_input_ids, all_attention_masks, all_labels = [], [], []
        skipped = 0

        for _, row in split_df.iterrows():
            user_content = prompt_template.replace("{sentence}", str(row["sentence"]))
            target_json = format_target_json(row.to_dict())

            messages = [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": target_json},
            ]

            # Full chat sequence: user + assistant
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            # Prompt portion ending with the assistant header (for masking)
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False, add_generation_prompt=True,
            )

            full_ids   = tokenizer(full_text,   add_special_tokens=False,
                                   truncation=True, max_length=max_length)["input_ids"]
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

            prompt_len = len(prompt_ids)
            if prompt_len >= len(full_ids):
                # Response was truncated away entirely — skip this example
                skipped += 1
                continue

            labels = [-100] * prompt_len + full_ids[prompt_len:]

            all_input_ids.append(full_ids)
            all_attention_masks.append([1] * len(full_ids))
            all_labels.append(labels)

        if skipped:
            print(f"  WARNING: {skipped} examples skipped (response truncated "
                  f"by max_length={max_length})")

        return Dataset.from_dict({
            "input_ids":      all_input_ids,
            "attention_mask": all_attention_masks,
            "labels":         all_labels,
        })

    return DatasetDict({
        "train":      make_hf_dataset(train_df),
        "validation": make_hf_dataset(val_df),
        "test":       make_hf_dataset(test_df),
    })


def build_multitask_dataset(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    label_maps: Dict[str, Dict[str, int]],
    label_fields: Optional[List[str]] = None,
    max_length: int = 256,
) -> Dataset:
    """
    Build a dataset with multiple label columns for multi-task learning.
    Each sample gets an integer label for every requested field.

    Args:
        df: Labelled DataFrame with 'sentence' and abbreviated label columns.
        tokenizer: Pre-loaded tokenizer.
        label_maps: Full maps dict from create_label_maps() —
                    {field: {label_str: int}}.
        label_fields: Which fields to include (defaults to all in label_maps).
        max_length: Maximum token sequence length.
    """
    if label_fields is None:
        label_fields = list(label_maps.keys())

    df = df.copy()
    for field in label_fields:
        col = f"{field}_label"
        if field in df.columns:
            df[col] = df[field].map(label_maps[field]).fillna(-1).astype(int)
        else:
            df[col] = -1

    texts = df["sentence"].tolist()
    encodings = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt",
    )

    out = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    }
    for field in label_fields:
        out[f"{field}_label"] = df[f"{field}_label"].tolist()

    return Dataset.from_dict(out)
