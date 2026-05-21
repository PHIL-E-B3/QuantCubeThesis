"""
count_prompt_tokens.py
----------------------
Counts exact tokens for all prompt files using the Llama 3.1 tokenizer.

Usage:
    python scripts/count_prompt_tokens.py --token YOUR_HF_TOKEN
    python scripts/count_prompt_tokens.py --token YOUR_HF_TOKEN --prompt P5_v11.txt
"""

import argparse
from pathlib import Path

ROOT        = Path(__file__).parent.parent
PROMPTS_DIR = ROOT / "prompts"
MODEL_ID    = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def count_tokens(tokenizer, prompt_text: str) -> int:
    messages = [{"role": "user", "content": prompt_text.replace("{sentence}", "PLACEHOLDER_SENTENCE")}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer(formatted, add_special_tokens=False)["input_ids"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",  type=str, required=True, help="HuggingFace access token")
    parser.add_argument("--prompt", type=str, default=None,  help="Single prompt file to count (default: all)")
    args = parser.parse_args()

    from huggingface_hub import login
    from transformers import AutoTokenizer

    login(token=args.token, add_to_git_credential=False)
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print()

    if args.prompt:
        files = [PROMPTS_DIR / args.prompt]
    else:
        files = sorted(PROMPTS_DIR.glob("*.txt"))

    results = []
    for p in files:
        try:
            text   = p.read_text(encoding="utf-8")
            tokens = count_tokens(tokenizer, text)
            words  = len(text.split())
            results.append((tokens, words, p.name))
        except Exception as e:
            print(f"  ERROR {p.name}: {e}")

    results.sort()
    print(f"{'Tokens':>8}  {'Words':>6}  {'Fits in':>10}  Prompt")
    print("-" * 70)
    for tokens, words, name in results:
        if tokens < 1280:   fits = "1280"
        elif tokens < 2048: fits = "2048"
        elif tokens < 4096: fits = "4096"
        else:               fits = ">4096"
        print(f"{tokens:>8}  {words:>6}  {fits:>10}  {name}")


if __name__ == "__main__":
    main()
