"""
Active Learning Script
======================
Usage:
    # Generate candidates for labelling (cycle 1):
    python scripts/active_learn.py --config configs/default.yaml --cycle 1 --select

    # After labelling, integrate and retrain:
    python scripts/active_learn.py --config configs/default.yaml --cycle 1 --integrate --retrain
"""

import argparse
import sys
import os
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.active_learning import ActiveLearner
from src.training.qlora_trainer import (
    load_model_and_tokenizer,
    get_lora_config,
    get_training_args,
    train,
)
from src.data.dataset import load_labels, create_label_maps, build_classification_dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="FOMC Active Learning")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--cycle", type=int, required=True, help="Active learning cycle number")
    parser.add_argument("--select", action="store_true", help="Select candidates for labelling")
    parser.add_argument("--integrate", action="store_true", help="Integrate new labels")
    parser.add_argument("--retrain", action="store_true", help="Retrain after integration")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to current best adapter (for uncertainty scoring)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    al_config = config["active_learning"]
    model_name = config["model"]["name"]

    # Primary label field for uncertainty sampling (e.g. "sen")
    primary = config.get("primary_label", "sen")
    all_maps = create_label_maps(config)
    label_map = all_maps[primary]
    id2label = {v: k for k, v in label_map.items()}

    learner = ActiveLearner(
        all_sentences_path=config["paths"]["unlabelled_pool"],
        labels_path=os.path.join(config["paths"]["data_labels"], "labels.json"),
        output_dir=os.path.join(config["paths"]["data_labels"], "active_learning"),
        query_size=al_config["query_size"],
        strategy=al_config["strategy"],
        holdout_crisis=al_config["holdout_crisis_episodes"],
    )

    if args.select:
        # Load current model for uncertainty scoring
        adapter_path = args.adapter_path or os.path.join(
            config["paths"]["model_output"], primary, "final_adapter"
        )

        lora_config = get_lora_config()
        model, tokenizer = load_model_and_tokenizer(
            model_name, len(label_map), label_map, id2label, lora_config,
        )

        # Load trained adapter weights if available
        from peft import PeftModel
        if os.path.exists(adapter_path):
            print(f"Loading adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model.base_model.model, adapter_path)

        candidates = learner.select_candidates(model, tokenizer)
        learner.export_for_labelling(candidates, args.cycle)

        if not learner.should_continue():
            print("\nActive learning appears to have converged.")

    if args.integrate:
        candidates_path = os.path.join(
            config["paths"]["data_labels"],
            "active_learning",
            f"candidates_cycle_{args.cycle:02d}.json",
        )
        learner.integrate_new_labels(candidates_path)

    if args.retrain:
        print("\n=== Retraining on expanded label set ===")
        df = load_labels(os.path.join(config["paths"]["data_labels"], "labels.json"))

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = build_classification_dataset(
            df, tokenizer, primary, label_map,
            max_length=config["model"]["max_seq_length"],
        )

        lora_cfg = config["lora"]
        lora_config = get_lora_config(
            r=lora_cfg["r"], lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
        )

        model, tokenizer = load_model_and_tokenizer(
            model_name, len(label_map), label_map, id2label, lora_config,
        )

        output_dir = os.path.join(config["paths"]["model_output"], f"al_cycle_{args.cycle}")
        train(model, tokenizer, dataset, output_dir=output_dir)


if __name__ == "__main__":
    main()
