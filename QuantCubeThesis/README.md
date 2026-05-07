# FOMC Sentiment Prediction Pipeline

Predicting Federal Reserve policy stance at *t+1* from FOMC statement text at *t*, using QLoRA fine-tuned open-source LLMs.

## Project Structure

```
├── configs/
│   └── default.yaml          # All hyperparameters & label taxonomy
├── src/
│   ├── data/
│   │   ├── fomc_parser.py     # FOMC statement → sentence decomposition
│   │   ├── dataset.py         # HuggingFace Dataset builder
│   │   └── preprocessing.py   # Text cleaning utilities
│   ├── training/
│   │   ├── qlora_trainer.py   # QLoRA fine-tuning (PEFT + bitsandbytes)
│   │   ├── hyperparameter_search.py  # Optuna integration
│   │   └── active_learning.py # Uncertainty-based active learning loop
│   ├── inference/
│   │   └── predict.py         # Distributional inference & Taylor rule input
│   └── evaluation/
│       └── metrics.py         # Classification metrics, calibration, plots
├── scripts/
│   ├── train.py               # Main training entry point
│   ├── active_learn.py        # Active learning workflow
│   └── evaluate.py            # Model evaluation
├── data/
│   ├── raw/                   # Raw FOMC statement .txt files
│   ├── labels/                # Hand-labelled CSVs
│   └── processed/             # Parsed sentence CSVs
├── models/                    # Saved LoRA adapters & checkpoints
└── notebooks/                 # Exploration & analysis
```

## Setup

### Windows
```
git clone https://github.com/PHIL-E-B3/QuantCubeThesis.git
cd QuantCubeThesis
setup_env.bat
```

### Mac / Linux
```
git clone https://github.com/PHIL-E-B3/QuantCubeThesis.git
cd QuantCubeThesis
chmod +x setup_env.sh && ./setup_env.sh
```

## Usage

### 1. Parse FOMC Statements
Place raw `.txt` files (named `YYYY-MM-DD.txt`) in `data/raw/`, then:
```bash
python -m src.data.fomc_parser data/raw data/processed/fomc_sentences.csv
```

### 2. Train (single run with defaults)
```bash
python scripts/train.py --config configs/default.yaml --baseline
```

### 3. Hyperparameter Search (Optuna)
```bash
python scripts/train.py --config configs/default.yaml --optuna
```

### 4. Active Learning
```bash
# Select uncertain sentences for labelling:
python scripts/active_learn.py --config configs/default.yaml --cycle 1 --select

# After labelling the exported CSV, integrate & retrain:
python scripts/active_learn.py --config configs/default.yaml --cycle 1 --integrate --retrain
```

### 5. Evaluate
```bash
python scripts/evaluate.py --config configs/default.yaml \
    --adapter-path models/forward_guidance/final_adapter
```

## Hardware Requirements

Tested on:
- **GPU**: NVIDIA RTX 4070 Ti Super (16GB VRAM)
- **RAM**: 32GB
- **Storage**: ~15GB for model weights + data

QLoRA 4-bit quantization makes this feasible on consumer GPUs. Peak VRAM usage is ~12-14GB during training.

## Label Taxonomy

**Forward Guidance**: odyssean_hawkish, odyssean_dovish, delphic_hawkish, delphic_dovish, neutral

**Economic Sentiment**: topic (inflation, labor, gdp_output, financial_conditions) × intensity (strongly_negative → strongly_positive)

See `configs/default.yaml` for full specification.
