import json
import os
import argilla as rg
from pathlib import Path

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
ARGILLA_URL     = os.getenv("ARGILLA_API_URL", "http://localhost:6900")
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY", "owner.apikey")

SEED_DIR = Path(__file__).parent.parent / "data" / "initial_training_seed_sentences"

LABEL_SCHEMA = {
    "top": ["inflation", "employment", "economic_activity", "macro",
            "financial_conditions", "monetary_policy", "no_topic", "boilerplate"],
    "ten": ["backward", "present", "forward", "hypothetical", "none"],
    "sen": ["-2", "-1", "0", "1", "2"],
    "dir": ["very hawkish", "hawkish", "neutral", "dovish", "very dovish"],
    "com": ["none", "conditional", "unconditional"],
    "hor": ["na", "short-term", "long-term"],
    "con": ["inflation", "employment", "economic_activity", "macro", "financial_conditions"],
    "dom": ["inflation", "employment", "economic_activity", "macro", "financial_conditions"],
    "ris": ["na", "skewed_downside", "symmetric", "skewed_upside"],
    "wid": ["none", "contested", "elevated"],
}

LABEL_TITLES = {
    "top": "Topic",
    "ten": "Tense",
    "sen": "Sentiment  (-2 very dovish → +2 very hawkish)",
    "dir": "Direction",
    "com": "Commitment",
    "hor": "Horizon",
    "con": "Condition Referenced",
    "dom": "Dominant Topic",
    "ris": "Risk Balance",
    "wid": "Width / Uncertainty",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

MULTILABEL_QUESTIONS = {"top", "con"}
REQUIRED_QUESTIONS   = {"top"}          # Argilla needs at least one; Topic applies to every sentence

def build_settings():
    fields = [
        rg.TextField(name="sentence",         title="Sentence",                     required=True),
        rg.TextField(name="context_question", title="Reporter Question (Q&A only)", required=False),
    ]
    questions = []
    for key, vals in LABEL_SCHEMA.items():
        required = key in REQUIRED_QUESTIONS
        if key in MULTILABEL_QUESTIONS:
            questions.append(
                rg.MultiLabelQuestion(name=key, title=LABEL_TITLES[key], labels=vals, required=required)
            )
        else:
            questions.append(
                rg.LabelQuestion(name=key, title=LABEL_TITLES[key], labels=vals, required=required)
            )
    metadata = [
        rg.TermsMetadataProperty(name="doc_type", title="Doc Type"),
        rg.TermsMetadataProperty(name="date",     title="Date"),
        rg.TermsMetadataProperty(name="source",   title="Source"),
    ]
    return rg.Settings(fields=fields, questions=questions, metadata=metadata)


def to_records(data):
    records = []
    for r in data:
        fields = {"sentence": r["sentence"]}
        cq = r.get("context_question")
        if cq:
            fields["context_question"] = cq
        records.append(rg.Record(
            fields=fields,
            metadata={
                "doc_type": r.get("doc_type", "unknown"),
                "date":     str(r.get("date", "unknown")),
                "source":   r.get("source", "unknown"),
            },
            id=r.get("id"),
        ))
    return records


def push_dataset(client, name, data):
    # Delete existing dataset if it exists so we start clean
    existing = client.datasets(name=name)
    if existing:
        existing.delete()
        print(f"  -> Deleted existing '{name}'")

    settings = build_settings()
    dataset = rg.Dataset(name=name, settings=settings, client=client)
    dataset.create()
    records = to_records(data)
    dataset.records.log(records)
    print(f"  -> '{name}': {len(records)} records pushed")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Connecting to Argilla at {ARGILLA_URL} ...")
    client = rg.Argilla(api_url=ARGILLA_URL, api_key=ARGILLA_API_KEY)
    print("Connected.\n")

    # final-press: prepared remarks (50) + Q&A answers (100)
    press_data = []
    for fname in ["seed_press_conference_prepared.json", "seed_press_conference_qa.json"]:
        with open(SEED_DIR / fname, encoding="utf-8") as f:
            press_data.extend(json.load(f))
    print(f"Pushing {len(press_data)} press conference records ...")
    push_dataset(client, "final-press", press_data)

    # final-speeches: speeches (150)
    with open(SEED_DIR / "seed_speech.json", encoding="utf-8") as f:
        speech_data = json.load(f)
    print(f"\nPushing {len(speech_data)} speech records ...")
    push_dataset(client, "final-speeches", speech_data)

    print(f"\nAll done. Open {ARGILLA_URL} to start labelling.")
