"""
Distributional Inference
========================
Multi-field inference for FOMC sentence classification.

For each sentence, loads one adapter per label field and extracts
the full softmax distribution.  Ordinal fields (sen, ris, wid)
also produce a continuous score via expected-value weighting,
which feeds into the Taylor-rule input vector.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from scipy.stats import entropy as scipy_entropy


# ── Ordinal weights for continuous scoring ──────────────────────
# Maps each label (as string) to a numeric position on the ordinal scale.
# Only fields with meaningful ordinal structure are included.
ORDINAL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "sen": {"-2": -2.0, "-1": -1.0, "0": 0.0, "1": 1.0, "2": 2.0},
    "ris": {
        "skewed_downside": -1.0, "na": 0.0,
        "symmetric": 0.0, "skewed_upside": 1.0,
    },
    "wid": {"none": 0.0, "contested": 1.0, "elevated": 2.0},
}


@dataclass
class FieldPrediction:
    """Prediction for a single label field on a single sentence."""
    field: str
    probabilities: Dict[str, float]
    predicted_label: str
    confidence: float
    entropy: float
    continuous_score: Optional[float] = None   # only for ordinal fields


@dataclass
class SentencePrediction:
    """All field-level predictions for one sentence."""
    sentence_id: str
    text: str
    fields: Dict[str, FieldPrediction] = field(default_factory=dict)


class FOMCPredictor:
    """
    Runs inference on FOMC sentences across multiple label fields.

    Each field has its own LoRA adapter fine-tuned on that classification
    task.  The predictor loads adapters lazily and caches them.
    """

    def __init__(
        self,
        base_model_name: str,
        adapter_dir: str,
        label_maps: Dict[str, Dict[str, int]],
        fields: Optional[List[str]] = None,
        max_length: int = 256,
        device: Optional[str] = None,
    ):
        """
        Args:
            base_model_name: HuggingFace model identifier.
            adapter_dir: Root directory containing per-field adapters,
                         e.g. ``models/sen/final_adapter``.
            label_maps: ``{field: {label_str: int}}`` from create_label_maps().
            fields: Which fields to predict (defaults to all in label_maps).
            max_length: Max sequence length for tokenization.
            device: Device override (auto-detected if None).
        """
        self.base_model_name = base_model_name
        self.adapter_dir = adapter_dir
        self.label_maps = label_maps
        self.id2labels = {
            f: {v: k for k, v in lm.items()} for f, lm in label_maps.items()
        }
        self.fields = fields or list(label_maps.keys())
        self.max_length = max_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Tokenizer (shared across fields)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache: field -> loaded model
        self._models: Dict[str, PeftModel] = {}

    # ── Model loading ──────────────────────────────────────────

    def _get_bnb_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def _load_field_model(self, field_name: str):
        """Load the base model + field-specific adapter."""
        import os

        id2label = self.id2labels[field_name]
        label2id = self.label_maps[field_name]
        num_labels = len(label2id)

        adapter_path = os.path.join(self.adapter_dir, field_name, "final_adapter")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"No adapter found at {adapter_path} for field '{field_name}'"
            )

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            quantization_config=self._get_bnb_config(),
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        self._models[field_name] = model

    def _get_model(self, field_name: str):
        if field_name not in self._models:
            self._load_field_model(field_name)
        return self._models[field_name]

    # ── Single-sentence inference ──────────────────────────────

    def predict_field(
        self, text: str, field_name: str,
    ) -> FieldPrediction:
        """Predict a single field for a single sentence."""
        model = self._get_model(field_name)
        id2label = self.id2labels[field_name]

        encodings = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(model.device) for k, v in encodings.items()}

        with torch.no_grad():
            logits = model(**encodings).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        prob_dict = {id2label[i]: float(p) for i, p in enumerate(probs)}
        pred_idx = int(np.argmax(probs))
        pred_label = id2label[pred_idx]
        confidence = float(probs[pred_idx])
        ent = float(scipy_entropy(probs))

        # Continuous score for ordinal fields
        cont_score = None
        if field_name in ORDINAL_WEIGHTS:
            weights = ORDINAL_WEIGHTS[field_name]
            cont_score = sum(
                prob_dict.get(lbl, 0.0) * w for lbl, w in weights.items()
            )

        return FieldPrediction(
            field=field_name,
            probabilities=prob_dict,
            predicted_label=pred_label,
            confidence=confidence,
            entropy=ent,
            continuous_score=cont_score,
        )

    def predict_sentence(
        self,
        text: str,
        sentence_id: str = "",
        fields: Optional[List[str]] = None,
    ) -> SentencePrediction:
        """Predict all requested fields for a single sentence."""
        fields = fields or self.fields
        pred = SentencePrediction(sentence_id=sentence_id, text=text)
        for f in fields:
            pred.fields[f] = self.predict_field(text, f)
        return pred

    # ── Statement-level aggregation ────────────────────────────

    def predict_statement(
        self,
        sentences: List[Dict[str, str]],
        fields: Optional[List[str]] = None,
    ) -> Dict:
        """
        Predict all fields for every sentence in a statement,
        then aggregate to statement level.

        Args:
            sentences: List of dicts with ``id`` and ``sentence`` keys.
            fields: Which fields to predict (defaults to self.fields).

        Returns:
            Dict with ``sentences`` (list) and ``aggregate`` (dict).
        """
        fields = fields or self.fields
        results: List[SentencePrediction] = []

        for s in sentences:
            pred = self.predict_sentence(
                s["sentence"], s.get("id", ""), fields=fields,
            )
            results.append(pred)

        aggregate = self._aggregate_statement(results, fields)

        return {
            "sentences": [self._serialise_prediction(r) for r in results],
            "aggregate": aggregate,
            "n_sentences": len(results),
        }

    @staticmethod
    def _serialise_prediction(pred: SentencePrediction) -> Dict:
        out = {"sentence_id": pred.sentence_id, "text": pred.text}
        for f, fp in pred.fields.items():
            out[f] = {
                "predicted": fp.predicted_label,
                "confidence": fp.confidence,
                "entropy": fp.entropy,
                "probabilities": fp.probabilities,
            }
            if fp.continuous_score is not None:
                out[f]["continuous_score"] = fp.continuous_score
        return out

    def _aggregate_statement(
        self,
        preds: List[SentencePrediction],
        fields: List[str],
    ) -> Dict:
        """Compute statement-level aggregates for each field."""
        agg: Dict = {}
        for f in fields:
            entropies = [p.fields[f].entropy for p in preds if f in p.fields]
            confidences = [p.fields[f].confidence for p in preds if f in p.fields]

            field_agg = {
                "mean_entropy": float(np.mean(entropies)) if entropies else None,
                "mean_confidence": float(np.mean(confidences)) if confidences else None,
            }

            # Label distribution (fraction of sentences getting each label)
            label_counts: Dict[str, int] = {}
            for p in preds:
                if f in p.fields:
                    lbl = p.fields[f].predicted_label
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
            n = sum(label_counts.values()) or 1
            field_agg["label_distribution"] = {
                lbl: cnt / n for lbl, cnt in label_counts.items()
            }

            # Continuous score aggregates for ordinal fields
            if f in ORDINAL_WEIGHTS:
                scores = [
                    p.fields[f].continuous_score
                    for p in preds
                    if f in p.fields and p.fields[f].continuous_score is not None
                ]
                if scores:
                    field_agg["mean_score"] = float(np.mean(scores))
                    field_agg["median_score"] = float(np.median(scores))
                    field_agg["std_score"] = float(np.std(scores))

            agg[f] = field_agg
        return agg

    # ── Taylor-rule input vector ───────────────────────────────

    def generate_taylor_input(
        self,
        statement_results: Dict,
    ) -> Dict[str, float]:
        """
        Convert statement-level aggregates into a Taylor-rule input vector.

        Returns dict with:
            sentiment_score   -- mean continuous score on 'sen'
            risk_balance      -- mean continuous score on 'ris'
            uncertainty_width -- mean continuous score on 'wid'
            sentiment_entropy -- mean prediction entropy on 'sen'
            dispersion        -- std of sentence-level 'sen' scores
        """
        agg = statement_results["aggregate"]

        def _safe(field_name: str, key: str, default: float = 0.0) -> float:
            return agg.get(field_name, {}).get(key, default) or default

        return {
            "sentiment_score": _safe("sen", "mean_score"),
            "risk_balance": _safe("ris", "mean_score"),
            "uncertainty_width": _safe("wid", "mean_score"),
            "sentiment_entropy": _safe("sen", "mean_entropy"),
            "dispersion": _safe("sen", "std_score"),
        }


def compute_continuous_score(
    probabilities: Dict[str, float],
    field_name: str,
) -> Optional[float]:
    """
    Standalone helper: compute the expected-value continuous score
    for an ordinal field given its probability distribution.
    """
    if field_name not in ORDINAL_WEIGHTS:
        return None
    weights = ORDINAL_WEIGHTS[field_name]
    return sum(probabilities.get(lbl, 0.0) * w for lbl, w in weights.items())
