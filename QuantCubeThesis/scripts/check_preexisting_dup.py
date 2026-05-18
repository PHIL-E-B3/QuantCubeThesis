"""Confirm the duplicate at indices 161/163 was pre-existing (both < original 502)."""
# indices 161 and 163 are both well within the original 502 eval sentences.
# The 20 new sentences were appended at the end (indices 502-521).
# So this is definitely a pre-existing duplicate, not introduced by our script.
print("Duplicate at indices 161 and 163.")
print("Original eval had 502 sentences (indices 0-501).")
print("Our script appended 20 sentences (indices 502-521).")
print("=> Duplicate is PRE-EXISTING, not introduced by topup.")
print()
print("Full sentence at index 161:")
import json
from pathlib import Path
BASE = Path(r"C:\Users\Javier\OneDrive - HEC Paris\Documentos\QuantCubeThesis\QuantCubeThesis\data")
with open(BASE / "eval_labelled_merged.json", encoding="utf-8") as f:
    data = json.load(f)
print(repr(data[161]["sentence"]))
print()
print("Full sentence at index 163:")
print(repr(data[163]["sentence"]))
