"""
Dataset Builder
===============
Converts labelled FOMC sentences into HuggingFace Datasets
ready for QLoRA fine-tuning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


def load_labels(labels_path: str) -> pd.DataFrame:
    """
    Load hand-labelled data from a JSON file.

    Required fields per record: id, sentence
    Label fields (abbreviated): top, ten, sen, dir, com, hor, con, dom, ris, wid
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
    "dir": "direction",
    "com": "commitment",
    "hor": "horizon",
    "con": "condition_referenced",
    "dom": "dominant_topic",
    "ris": "risk_balance",
    "wid": "width",
}


def create_label_maps(config: dict) -> Dict[str, Dict[str, int]]:
    """
    Build label-to-id mappings for all label dimensions from config.

    Returns:
        Dict mapping each abbreviated field name to its {label_str: int} map.
        e.g. {"sen": {"hawkish": 0, "dovish": 1, ...}, "top": {...}, ...}
    """
    return {
        field: {label: i for i, label in enumerate(values)}
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
) -> DatasetDict:
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
        DatasetDict with train/validation/test splits.
    """
    # Filter to rows that have a valid label
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

        encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        })

    return DatasetDict({
        "train": tokenize_split(train_df),
        "validation": tokenize_split(val_df),
        "test": tokenize_split(test_df),
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
