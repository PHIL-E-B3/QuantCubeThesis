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
    Load hand-labelled data.

    Expected CSV columns:
        sentence_id, text, forward_guidance, fg_uncertainty,
        econ_topic, econ_intensity, [optional: notes]

    forward_guidance:  odyssean_hawkish | odyssean_dovish |
                       delphic_hawkish  | delphic_dovish  | neutral
    fg_uncertainty:    low | medium | high
    econ_topic:        inflation | labor | gdp_output |
                       financial_conditions | none
    econ_intensity:    strongly_negative | negative | neutral |
                       positive | strongly_positive | none
    """
    df = pd.read_csv(labels_path)

    required_cols = ["sentence_id", "text", "forward_guidance"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def create_label_maps(config: dict) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Create label-to-id mappings from config."""
    fg_labels = config["labels"]["forward_guidance"]
    fg_map = {label: i for i, label in enumerate(fg_labels)}

    intensity_labels = config["labels"]["economic_sentiment"]["intensity"]
    intensity_map = {label: i for i, label in enumerate(intensity_labels)}

    return fg_map, intensity_map


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
        df: DataFrame with 'text' and label_column columns.
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
        texts = split_df["text"].tolist()
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
    fg_map: Dict[str, int],
    intensity_map: Dict[str, int],
    max_length: int = 256,
) -> Dataset:
    """
    Build a dataset with multiple label columns for multi-task learning.
    Each sample has both a forward_guidance label and an econ_intensity label.
    Useful for joint training or for comparing single-task vs multi-task.
    """
    df = df.copy()
    df["fg_label"] = df["forward_guidance"].map(fg_map).fillna(-1).astype(int)
    df["intensity_label"] = df.get("econ_intensity", pd.Series()).map(
        intensity_map
    ).fillna(-1).astype(int)

    texts = df["text"].tolist()
    encodings = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt",
    )

    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "fg_label": df["fg_label"].tolist(),
        "intensity_label": df["intensity_label"].tolist(),
    })
