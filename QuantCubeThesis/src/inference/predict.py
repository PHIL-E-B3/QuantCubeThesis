"""
Distributional Inference
========================
Extracts softmax probability distributions from the fine-tuned model.
The distribution captures:
- Directional sentiment (hawk vs dove)
- Asymmetry (skewness of the distribution)
- Uncertainty (entropy / width of the distribution)
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from scipy.stats import entropy as scipy_entropy


@dataclass
class SentimentDistribution:
    """The distributional output for a single sentence."""
    sentence_id: str
    text: str
    probabilities: Dict[str, float]     # {label: probability}
    predicted_label: str
    confidence: float                   # max probability
    entropy: float                      # Shannon entropy (uncertainty)
    hawk_dove_score: float              # Continuous score: -1 (dove) to +1 (hawk)
    asymmetry: float                    # Skewness of the distribution


class FOMCPredictor:
    """
    Runs inference on FOMC statements and returns distributional outputs.
    """

    def __init__(
        self,
        base_model_name: str,
        adapter_path: str,
        id2label: Dict[int, str],
        max_length: int = 256,
        device: Optional[str] = None,
    ):
        """
        Args:
            base_model_name: HuggingFace model name (e.g., 'mistralai/Mistral-7B-Instruct-v0.3').
            adapter_path: Path to the saved LoRA adapter.
            id2label: Mapping from label IDs to label names.
            max_length: Max sequence length for tokenization.
            device: Device to use (auto-detected if None).
        """
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.max_length = max_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        # Load base model + LoRA adapter
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(id2label),
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

        # Precompute hawk/dove weights for the continuous score
        self._compute_hd_weights()

    def _compute_hd_weights(self):
        """
        Assign numeric sentiment weights to each label for the continuous score.
        Positive = hawkish/tightening, Negative = dovish/easing, 0 = neutral.

        Update this map once you have finalised your label values in the config.
        Any label not listed here defaults to 0.0 (neutral).
        """
        weight_map: Dict[str, float] = {
            # --- Fill in once your 'sen' label values are defined ---
            # Example (replace with your actual values):
            # "hawkish": 1.0,
            # "neutral": 0.0,
            # "dovish": -1.0,
        }
        self.hd_weights = {}
        for idx, label in self.id2label.items():
            self.hd_weights[idx] = weight_map.get(label, 0.0)

    def predict_sentence(self, text: str, sentence_id: str = "") -> SentimentDistribution:
        """Get the full distributional output for a single sentence."""
        encodings = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.model.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        # Build probability dict
        prob_dict = {self.id2label[i]: float(p) for i, p in enumerate(probs)}

        # Predicted label & confidence
        pred_idx = int(np.argmax(probs))
        pred_label = self.id2label[pred_idx]
        confidence = float(probs[pred_idx])

        # Entropy (uncertainty measure)
        ent = float(scipy_entropy(probs))

        # Hawk/dove continuous score (weighted sum)
        hd_score = sum(probs[i] * self.hd_weights.get(i, 0) for i in range(len(probs)))

        # Asymmetry: skewness of the probability distribution
        # Positive skew = more weight on hawkish tail
        weights = np.array([self.hd_weights.get(i, 0) for i in range(len(probs))])
        mean_w = np.sum(probs * weights)
        var_w = np.sum(probs * (weights - mean_w) ** 2)
        std_w = np.sqrt(var_w) if var_w > 0 else 1e-8
        skew = float(np.sum(probs * ((weights - mean_w) / std_w) ** 3))

        return SentimentDistribution(
            sentence_id=sentence_id,
            text=text,
            probabilities=prob_dict,
            predicted_label=pred_label,
            confidence=confidence,
            entropy=ent,
            hawk_dove_score=float(hd_score),
            asymmetry=skew,
        )

    def predict_statement(
        self,
        sentences: List[Dict[str, str]],
    ) -> Dict:
        """
        Predict distributional output for an entire FOMC statement.

        Args:
            sentences: List of dicts with 'sentence_id' and 'text' keys.

        Returns:
            Dict with sentence-level results + aggregate statement-level metrics.
        """
        results = []
        for s in sentences:
            dist = self.predict_sentence(s["sentence"], s.get("id", ""))
            results.append(dist)

        # ── Aggregate to statement level ────────────────────────
        hd_scores = [r.hawk_dove_score for r in results]
        entropies = [r.entropy for r in results]

        aggregate = {
            "mean_hawk_dove_score": float(np.mean(hd_scores)),
            "median_hawk_dove_score": float(np.median(hd_scores)),
            "std_hawk_dove_score": float(np.std(hd_scores)),   # Dispersion
            "mean_entropy": float(np.mean(entropies)),         # Avg uncertainty
            "max_entropy": float(np.max(entropies)),           # Most uncertain sentence
            "hawk_fraction": float(np.mean([s > 0 for s in hd_scores])),
            "dove_fraction": float(np.mean([s < 0 for s in hd_scores])),
            "n_sentences": len(results),
        }

        return {
            "sentences": [
                {
                    "sentence_id": r.sentence_id,
                    "text": r.text,
                    "predicted_label": r.predicted_label,
                    "confidence": r.confidence,
                    "entropy": r.entropy,
                    "hawk_dove_score": r.hawk_dove_score,
                    "asymmetry": r.asymmetry,
                    "probabilities": r.probabilities,
                }
                for r in results
            ],
            "aggregate": aggregate,
        }

    def generate_taylor_input(
        self,
        statement_results: Dict,
    ) -> Dict[str, float]:
        """
        Convert statement-level aggregate into a Taylor rule input vector.

        Returns dict with:
            - sentiment_score: continuous hawk/dove indicator [-1, 1]
            - uncertainty: entropy-based uncertainty measure [0, ∞)
            - asymmetry: directional skew of sentiment
            - dispersion: std of sentence-level scores (disagreement within statement)
        """
        agg = statement_results["aggregate"]

        return {
            "sentiment_score": agg["mean_hawk_dove_score"],
            "uncertainty": agg["mean_entropy"],
            "asymmetry": float(np.mean([
                s["asymmetry"] for s in statement_results["sentences"]
            ])),
            "dispersion": agg["std_hawk_dove_score"],
        }
