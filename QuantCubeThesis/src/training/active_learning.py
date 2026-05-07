"""
Active Learning Loop
====================
Iteratively selects the most informative unlabelled sentences
for human annotation, using uncertainty sampling from the
fine-tuned QLoRA model.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from transformers import AutoTokenizer
from scipy.stats import entropy as scipy_entropy


class ActiveLearner:
    """
    Manages the active learning loop:
    1. Train on current labelled set
    2. Score unlabelled sentences by uncertainty
    3. Select top-k most uncertain for human labelling
    4. Integrate new labels, repeat
    """

    def __init__(
        self,
        all_sentences_path: str,
        labels_path: str,
        output_dir: str,
        query_size: int = 80,
        strategy: str = "entropy",
        holdout_crisis: bool = True,
    ):
        """
        Args:
            all_sentences_path: CSV with all parsed FOMC sentences.
            labels_path: CSV with current hand-labelled data.
            output_dir: Where to save candidates and logs.
            query_size: How many sentences to select per cycle.
            strategy: Uncertainty measure — 'entropy', 'margin', or 'least_confidence'.
            holdout_crisis: Whether to flag GFC/COVID sentences separately.
        """
        self.all_sentences = pd.read_csv(all_sentences_path)
        self.labels_path = labels_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.query_size = query_size
        self.strategy = strategy
        self.holdout_crisis = holdout_crisis

        self._load_labels()
        self.cycle_log = []

    def _load_labels(self):
        """Load current labels and compute labelled/unlabelled split."""
        if os.path.exists(self.labels_path):
            self.labelled = pd.read_csv(self.labels_path)
            labelled_ids = set(self.labelled["sentence_id"])
        else:
            self.labelled = pd.DataFrame()
            labelled_ids = set()

        self.unlabelled = self.all_sentences[
            ~self.all_sentences["sentence_id"].isin(labelled_ids)
        ].copy()

        print(f"Labelled: {len(self.labelled)} | Unlabelled: {len(self.unlabelled)}")

    def compute_uncertainty(
        self,
        model,
        tokenizer: AutoTokenizer,
        texts: List[str],
        max_length: int = 256,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Compute uncertainty scores for a list of texts.

        Returns:
            Array of uncertainty scores (higher = more uncertain).
        """
        model.eval()
        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encodings = {k: v.to(model.device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = model(**encodings)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        all_probs = np.concatenate(all_probs, axis=0)

        # Compute uncertainty based on strategy
        if self.strategy == "entropy":
            # Shannon entropy of the softmax distribution
            scores = scipy_entropy(all_probs, axis=1)
        elif self.strategy == "margin":
            # 1 - (top_prob - second_prob): smaller margin = more uncertain
            sorted_probs = np.sort(all_probs, axis=1)[:, ::-1]
            scores = 1.0 - (sorted_probs[:, 0] - sorted_probs[:, 1])
        elif self.strategy == "least_confidence":
            # 1 - max_prob
            scores = 1.0 - np.max(all_probs, axis=1)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return scores

    def select_candidates(
        self,
        model,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ) -> pd.DataFrame:
        """
        Score all unlabelled sentences and return top-k candidates.

        Returns:
            DataFrame of candidates sorted by uncertainty (descending).
        """
        if len(self.unlabelled) == 0:
            print("No unlabelled sentences remaining!")
            return pd.DataFrame()

        texts = self.unlabelled["text"].tolist()
        scores = self.compute_uncertainty(model, tokenizer, texts, max_length)

        self.unlabelled["uncertainty_score"] = scores

        # Optionally flag crisis episodes
        if self.holdout_crisis:
            crisis_mask = self.unlabelled["regime_hint"] == "crisis"
            crisis_candidates = self.unlabelled[crisis_mask].nlargest(
                min(10, crisis_mask.sum()), "uncertainty_score"
            )
            non_crisis = self.unlabelled[~crisis_mask].nlargest(
                self.query_size, "uncertainty_score"
            )
            candidates = pd.concat([non_crisis, crisis_candidates])
            candidates["is_crisis_episode"] = candidates["regime_hint"] == "crisis"
        else:
            candidates = self.unlabelled.nlargest(self.query_size, "uncertainty_score")

        return candidates

    def export_for_labelling(
        self,
        candidates: pd.DataFrame,
        cycle: int,
    ) -> str:
        """
        Export candidates to CSV for human annotation.

        The CSV has columns for the annotator to fill in:
        - forward_guidance
        - fg_uncertainty
        - econ_topic
        - econ_intensity
        - notes (optional)
        """
        export_df = candidates[[
            "sentence_id", "text", "statement_date", "chair_era",
            "regime_hint", "uncertainty_score",
        ]].copy()

        # Add empty columns for annotator
        export_df["forward_guidance"] = ""
        export_df["fg_uncertainty"] = ""
        export_df["econ_topic"] = ""
        export_df["econ_intensity"] = ""
        export_df["notes"] = ""

        filename = f"candidates_cycle_{cycle:02d}.csv"
        filepath = self.output_dir / filename
        export_df.to_csv(filepath, index=False)

        # Log this cycle
        self.cycle_log.append({
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
            "n_candidates": len(candidates),
            "mean_uncertainty": float(candidates["uncertainty_score"].mean()),
            "total_labelled": len(self.labelled),
            "total_unlabelled": len(self.unlabelled),
        })

        log_path = self.output_dir / "active_learning_log.json"
        with open(log_path, "w") as f:
            json.dump(self.cycle_log, f, indent=2)

        print(f"\nCycle {cycle}: exported {len(candidates)} candidates to {filepath}")
        print(f"  Mean uncertainty: {candidates['uncertainty_score'].mean():.4f}")
        print(f"  Labelled so far: {len(self.labelled)}")

        return str(filepath)

    def integrate_new_labels(self, new_labels_path: str):
        """
        After human annotation, integrate new labels into the labelled set.
        Call this with the path to the filled-in candidates CSV.
        """
        new = pd.read_csv(new_labels_path)

        # Filter to only rows that were actually labelled
        labelled_mask = new["forward_guidance"].notna() & (new["forward_guidance"] != "")
        new_labelled = new[labelled_mask]

        if len(new_labelled) == 0:
            print("No new labels found in file!")
            return

        # Append to master labels file
        if len(self.labelled) > 0:
            self.labelled = pd.concat([self.labelled, new_labelled], ignore_index=True)
        else:
            self.labelled = new_labelled.copy()

        self.labelled.to_csv(self.labels_path, index=False)
        self._load_labels()  # Refresh unlabelled set

        print(f"Integrated {len(new_labelled)} new labels. Total: {len(self.labelled)}")

    def should_continue(self, min_uncertainty_drop: float = 0.05) -> bool:
        """
        Heuristic to check if active learning has plateaued.
        If mean uncertainty hasn't dropped by at least `min_uncertainty_drop`
        over the last 2 cycles, we've likely converged.
        """
        if len(self.cycle_log) < 2:
            return True

        recent = self.cycle_log[-1]["mean_uncertainty"]
        previous = self.cycle_log[-2]["mean_uncertainty"]
        drop = previous - recent

        if drop < min_uncertainty_drop:
            print(f"Uncertainty drop ({drop:.4f}) below threshold "
                  f"({min_uncertainty_drop}). Consider stopping.")
            return False
        return True
