"""
Evaluation Script
=================
Usage:
    python scripts/evaluate.py --config configs/default.yaml --task sen --adapter-dir models
    python scripts/evaluate.py --config configs/default.yaml --task dir --adapter-dir models
"""

import argparse
import sys
import os
import yaml
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import load_labels, create_label_maps, build_classification_dataset
from src.evaluation.metrics import full_classification_report, compare_baseline_vs_finetuned
from src.inference.predict import FOMCPredictor
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate FOMC Sentiment Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--adapter-dir", type=str, required=True,
                        help="Root directory containing per-field adapters (e.g. models/)")
    parser.add_argument("--task", type=str, default="sen",
                        choices=["top", "ten", "sen", "com", "hor", "ris", "wid"])
    parser.add_argument("--output-dir", type=str, default="models/evaluation")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    all_maps = create_label_maps(config)

    if args.task not in all_maps:
        raise ValueError(f"Task '{args.task}' has no values defined in config labels.")

    label_map = all_maps[args.task]
    id2label = {v: k for k, v in label_map.items()}
    label_names = [id2label[i] for i in range(len(id2label))]

    # Load test data
    labels_path = os.path.join(config["paths"]["data_labels"], "labels.json")
    df = load_labels(labels_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_classification_dataset(
        df, tokenizer, args.task, label_map,
        max_length=config["model"]["max_seq_length"],
    )

    # Load predictor (single-field evaluation)
    predictor = FOMCPredictor(
        base_model_name=model_name,
        adapter_dir=args.adapter_dir,
        label_maps={args.task: label_map},
        fields=[args.task],
        max_length=config["model"]["max_seq_length"],
    )

    # Get predictions on test set
    test_data = dataset["test"]
    y_true = np.array(test_data["labels"])

    all_probs = []
    for i in range(len(test_data)):
        text = tokenizer.decode(test_data["input_ids"][i], skip_special_tokens=True)
        fp = predictor.predict_field(text, args.task)
        probs = [fp.probabilities.get(label_names[j], 0.0) for j in range(len(label_names))]
        all_probs.append(probs)

    y_probs = np.array(all_probs)
    y_pred = np.argmax(y_probs, axis=-1)

    # Full report
    os.makedirs(args.output_dir, exist_ok=True)
    results = full_classification_report(
        y_true, y_pred, y_probs, label_names, args.output_dir,
    )

    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    serializable = {k: v for k, v in results.items() if k != "per_class"}
    with open(results_path, "w") as outf:
        json.dump(serializable, outf, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
