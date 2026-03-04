# Mental Health Text Classification Using Fine-Tuned RoBERTa-Large

A deep learning pipeline that classifies text into **7 mental health categories** using a fine-tuned **RoBERTa-large** transformer with focal loss, triple pooling, and temperature scaling.

## Categories

| # | Class |
|---|-------|
| 1 | Normal |
| 2 | Depression |
| 3 | Suicidal |
| 4 | Anxiety |
| 5 | Bipolar |
| 6 | Stress |
| 7 | Personality Disorder |

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **95.58%** |
| Macro F1 | **93.48%** |
| Weighted F1 | **95.59%** |
| Best Epoch | 19 / 20 |
| Training Time | ~12.5 hours |

## Key Techniques

- **Partial Fine-Tuning** — Bottom 14 layers frozen; top 10 trained with layer-wise LR decay (γ = 0.9)
- **Triple Pooling** — CLS + Mean + Max pooling concatenated into a 3,072-dim representation
- **Focal Loss** — With label smoothing (ε = 0.05) and class-frequency weighting to handle imbalance
- **Temperature Scaling** — Post-training calibration for reliable confidence scores
- **Back-Translation Augmentation** — Synthetic samples for minority classes

## Architecture

```
Input Text
    │
    ▼
RoBERTa-large Tokenizer (BPE, max_len=256)
    │
    ▼
RoBERTa-large Encoder (24 layers, 1024 hidden)
  ├─ Layers 0–13: Frozen
  └─ Layers 14–23: Fine-tuned (LR decay)
    │
    ├── CLS token    ──┐
    ├── Mean pooling  ──┼── Concat → 3072-dim
    └── Max pooling   ──┘
    │
    ▼
Classification Head
  LayerNorm → Linear(3072→512) → GELU → Dropout(0.30)
            → Linear(512→128)  → GELU → Dropout(0.15)
            → Linear(128→7)
    │
    ▼
Temperature-Scaled Softmax → 7-class probabilities
```

## Project Structure

```
├── main.py                 # Training pipeline (fine-tuning, evaluation, saving)
├── app.py                  # Streamlit web application (inference UI)
├── download.py             # Dataset download utility
├── generate_architecture.py# Architecture diagram generation
├── requirements.txt        # Python dependencies
├── research_paper.tex      # LaTeX research paper
├── research_paper_v2.tex   # Updated paper version
├── augmented_data.csv      # Back-translation augmented data
├── Combined Data.csv       # Original dataset
├── training_log.txt        # Training output log
└── output/
    ├── best_weights.pt     # Trained model weights
    ├── model_meta.json     # Hyperparameters & training history
    └── tokenizer/          # Saved tokenizer files
```

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd Model(v2)
```

### 2. Create a virtual environment

```bash
python -m venv env
```

**Windows:**
```powershell
.\env\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python main.py
```

Trains the model from scratch. Outputs weights, tokenizer, and metadata to `output/`.

### Web Application

```bash
streamlit run app.py
```

Launches an interactive Streamlit app with the following pages:

| Page | Description |
|------|-------------|
| **Classify** | Single-text inference with confidence scores and probability charts |
| **Batch** | CSV upload / multi-line batch analysis |
| **Training** | Interactive training history & metrics visualization |
| **Architecture** | Model architecture visual explanation |
| **About** | Methodology, dataset info, and ethical considerations |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | `roberta-large` |
| Max Sequence Length | 256 |
| Trainable Layers | Top 10 / 24 |
| Layer LR Decay | 0.9 |
| Encoder LR | 2e-5 |
| Head LR | 1e-4 |
| Batch Size | 16 (effective 64 with grad accum) |
| Epochs | 20 |
| Dropout | 0.30 |
| Focal Loss γ | 2.0 |
| Label Smoothing ε | 0.05 |
| Weight Decay | 0.02 |
| Warmup Ratio | 6% |

## Dataset

The [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) dataset from Kaggle — 53,043 samples across 7 classes sourced from social media and mental health forums.

## Disclaimer

> **This project is a research and educational tool only.** It is **not** intended to replace professional mental health assessment or diagnosis. If you or someone you know is struggling, please reach out to a qualified mental health professional or crisis service.

## License

This project is for academic purposes.
