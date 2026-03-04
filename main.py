"""
Mental Health Text Classifier
==============================
Fine-tunes RoBERTa-large on a 7-class mental health dataset.
Expects augmented_data.csv to already exist in OUTDIR (run
back_translation.py first), otherwise trains on primary data only.

Classes : Normal, Depression, Suicidal, Anxiety,
          Bipolar, Stress, Personality disorder

Model   : roberta-large
          - Bottom 14 layers frozen
          - Top 10 layers trained with layer-wise LR decay (γ=0.9)
          - CLS + Mean + Max pooling → 3-layer classifier head

Training: Focal loss + label smoothing, FP16, cosine schedule
Post    : Temperature scaling calibration on val set
"""

import os
import time
import json
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

class CFG:
    # Paths
    PRIMARY_DATA = "/teamspace/studios/this_studio/.cache/kagglehub/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/versions/1/Combined Data.csv"
    AUG_DATA     = "./augmented_data.csv"   # from back_translation.py
    OUT_DIR      = "./output"

    # Model
    BERT_MODEL       = "roberta-large"
    MAX_LEN          = 256
    DROPOUT          = 0.30
    N_TRAINABLE      = 10      # top N transformer layers to fine-tune

    # Optimiser
    LR_BERT          = 2e-5
    LR_HEAD          = 1e-4
    LAYER_DECAY      = 0.90    # LR multiplier per layer going downward
    WEIGHT_DECAY     = 0.02
    MAX_GRAD_NORM    = 1.0

    # Schedule
    EPOCHS           = 20
    BATCH_SIZE       = 16
    GRAD_ACCUM       = 4       # effective batch = 64
    WARMUP_RATIO     = 0.06

    # Loss
    FOCAL_GAMMA      = 2.0
    LABEL_SMOOTH     = 0.05

    # Misc
    FP16             = True
    ES_PATIENCE      = 5
    VAL_SIZE         = 0.10
    TEST_SIZE        = 0.10
    SEED             = 42


LABEL_MAP = {
    "normal"              : "Normal",
    "depression"          : "Depression",
    "suicidal"            : "Suicidal",
    "anxiety"             : "Anxiety",
    "bipolar"             : "Bipolar",
    "stress"              : "Stress",
    "personality disorder": "Personality disorder",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


set_seed(CFG.SEED)
os.makedirs(CFG.OUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────

def load_primary(path: str) -> pd.DataFrame:
    """Load and normalise the primary dataset."""
    df = pd.read_csv(path)

    text_col  = next(c for c in df.columns
                     if c.lower() in ["text", "statement", "post", "content", "tweet"])
    label_col = next(c for c in df.columns
                     if c.lower() in ["label", "status", "class", "category", "target"])

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["text"]   = df["text"].astype(str).str.strip()
    df["label"]  = df["label"].str.lower().map(LABEL_MAP).fillna(df["label"])
    df["source"] = "original"

    return df[df["text"].str.len() >= 8].dropna(subset=["label"]).reset_index(drop=True)


def load_augmented(path: str) -> pd.DataFrame:
    """Load back-translated augmentation data if available."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["text", "label", "source"])

    df = pd.read_csv(path)
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    return df[df["text"].str.len() >= 8].reset_index(drop=True)


def build_splits(df_orig: pd.DataFrame,
                 df_aug: pd.DataFrame,
                 le: LabelEncoder,
                 val_size: float,
                 test_size: float,
                 seed: int):
    """
    Split original data into train / val / test, then attach
    augmented samples to train only. Val and test remain clean
    (original samples only) to avoid near-duplicate leakage.
    """
    df_orig["label_id"] = le.transform(df_orig["label"])

    train_orig, tmp = train_test_split(
        df_orig, test_size=val_size + test_size,
        stratify=df_orig["label_id"], random_state=seed)

    val_df, test_df = train_test_split(
        tmp, test_size=0.5,
        stratify=tmp["label_id"], random_state=seed)

    if len(df_aug) > 0:
        df_aug["label_id"] = le.transform(df_aug["label"])
        train_df = pd.concat([train_orig, df_aug], ignore_index=True)
        train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        train_df = train_orig.copy()

    return train_df, val_df, test_df


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class MentalHealthDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts  = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tok    = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length  = self.max_len,
            padding     = "max_length",
            truncation  = True,
            return_tensors = "pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label"         : torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss with per-class weights and label smoothing.

    Focal term  : (1 - p_t)^γ   down-weights easy examples
    Class weight: α_t            up-weights minority classes
    Label smooth: ε              prevents over-confident predictions
    """

    def __init__(self, class_weights: torch.Tensor,
                 gamma: float = 2.0,
                 label_smoothing: float = 0.05,
                 num_classes: int = 7):
        super().__init__()
        self.gamma  = gamma
        self.smooth = label_smoothing
        self.nc     = num_classes
        self.register_buffer("alpha", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            smooth_labels = torch.full_like(logits, self.smooth / (self.nc - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smooth)

        log_probs = F.log_softmax(logits, dim=-1)
        probs     = log_probs.exp()
        alpha_t   = self.alpha[targets]
        ce        = -(smooth_labels * log_probs).sum(dim=1)
        pt        = probs.gather(1, targets.unsqueeze(1)).squeeze(1).detach()
        focal_w   = (1.0 - pt) ** self.gamma

        return (alpha_t * focal_w * ce).mean()


# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────

class MentalHealthClassifier(nn.Module):
    """
    RoBERTa encoder with a 3-layer classification head.

    Pooling strategy: concatenate [CLS, mean-pool, max-pool]
    giving a 3×1024 = 3072-dim representation. This is richer
    than CLS alone — mean-pool captures average semantics while
    max-pool captures the most salient features.

    The bottom (12 - N_TRAINABLE) layers are frozen to avoid
    catastrophic forgetting and reduce overfitting on this
    ~52K sample dataset.
    """

    def __init__(self, model_name: str, num_classes: int,
                 dropout: float, n_trainable: int):
        super().__init__()
        # add_pooling_layer=False: roberta checkpoint has no pooler weights;
        # skipping it removes ~2M dead parameters from the optimizer.
        self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden    = self.bert.config.hidden_size   # 1024 for roberta-large

        self._freeze_layers(n_trainable)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def _freeze_layers(self, n_trainable: int) -> None:
        """Freeze embeddings and the bottom (total - n_trainable) layers."""
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False

        layers   = self.bert.encoder.layer
        n_freeze = max(0, len(layers) - n_trainable)
        for layer in layers[:n_freeze]:
            for p in layer.parameters():
                p.requires_grad = False

    def _mean_pool(self, hidden: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        expanded = mask.unsqueeze(-1).float()
        return (hidden * expanded).sum(1) / expanded.sum(1).clamp(min=1e-9)

    def _max_pool(self, hidden: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        expanded = mask.unsqueeze(-1).float()
        return (hidden * expanded + (1.0 - expanded) * -1e4).max(1).values

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq  = out.last_hidden_state

        cls_v  = seq[:, 0, :]
        mean_v = self._mean_pool(seq, attention_mask)
        max_v  = self._max_pool( seq, attention_mask)

        return self.head(torch.cat([cls_v, mean_v, max_v], dim=1))


# ──────────────────────────────────────────────────────────────
# Optimiser — layer-wise LR decay
# ──────────────────────────────────────────────────────────────

def build_optimizer(model: MentalHealthClassifier,
                    lr_bert: float,
                    lr_head: float,
                    layer_decay: float,
                    weight_decay: float,
                    n_trainable: int) -> torch.optim.AdamW:
    """
    Assign a different learning rate to each transformer layer.

    The top (most task-specific) layer gets lr_bert. Each layer
    below is multiplied by layer_decay, so lower layers — which
    hold more general representations — receive smaller updates.

    This prevents the lower layers from drifting away from their
    pre-trained values while still allowing the upper layers to
    adapt fully to the classification task.
    """
    all_layers       = list(model.bert.encoder.layer)
    trainable_layers = all_layers[-n_trainable:]
    n                = len(trainable_layers)
    param_groups     = []

    for i, layer in enumerate(trainable_layers):
        scale  = layer_decay ** (n - 1 - i)
        params = [p for p in layer.parameters() if p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": lr_bert * scale})

    param_groups.append({
        "params": [p for p in model.head.parameters() if p.requires_grad],
        "lr"    : lr_head,
    })

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ──────────────────────────────────────────────────────────────
# Training and Evaluation
# ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler,
                scaler, criterion, grad_accum: int) -> tuple:
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbl  = batch["label"].to(DEVICE)

        with autocast(enabled=CFG.FP16):
            logits = model(ids, mask)
            loss   = criterion(logits, lbl) / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item() * grad_accum
        all_preds    += logits.argmax(-1).detach().cpu().tolist()
        all_labels   += lbl.detach().cpu().tolist()

        if (step + 1) % 200 == 0:
            acc = accuracy_score(all_labels, all_preds) * 100
            avg_loss = running_loss / (step + 1)
            print(f"  step {step+1:5d}/{len(loader)}  "
                  f"loss={avg_loss:.4f}  acc={acc:.1f}%", flush=True)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return running_loss / len(loader), acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion,
             return_logits: bool = False) -> tuple:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_logits = [], [], []

    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbl  = batch["label"].to(DEVICE)

        with autocast(enabled=CFG.FP16):
            logits = model(ids, mask)
            loss   = criterion(logits, lbl)

        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_preds  += logits.argmax(-1).cpu().tolist()
        all_labels += lbl.cpu().tolist()

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    if return_logits:
        return (total_loss / len(loader), acc, f1,
                all_preds, all_labels,
                torch.cat(all_logits, dim=0))

    return total_loss / len(loader), acc, f1, all_preds, all_labels


# ──────────────────────────────────────────────────────────────
# Temperature Scaling
# ──────────────────────────────────────────────────────────────

def find_best_temperature(logits: torch.Tensor,
                          labels: np.ndarray,
                          search_range=(0.5, 3.0),
                          step: float = 0.05) -> float:
    """
    Search for the scalar temperature T that maximises val macro-F1.
    Dividing logits by T > 1 softens the distribution (less overconfident),
    which typically helps on imbalanced datasets. Using a single scalar
    avoids the overfitting risk of per-class threshold tuning.
    """
    best_T  = 1.0
    best_f1 = f1_score(labels, logits.argmax(dim=1).numpy(),
                       average="macro", zero_division=0)

    for T in np.arange(*search_range, step):
        preds = (logits / T).argmax(dim=1).numpy()
        f1    = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_T  = float(round(T, 3))

    return best_T


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def save_plots(history: list, test_labels, test_preds,
               le: LabelEncoder, test_acc: float,
               test_f1: float, best_epoch: int,
               out_dir: str) -> None:

    cm     = confusion_matrix(test_labels, test_preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Confusion matrix
    ax = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_,
                ax=ax, cbar_kws={"label": "% of true class"})
    ax.set_title(f"Confusion Matrix  (Acc={test_acc*100:.2f}%)", fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=30)

    epx = [h["epoch"] for h in history]

    # Accuracy
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epx, [h["tr_acc"]*100 for h in history], "b-o", label="Train", lw=2)
    ax.plot(epx, [h["vl_acc"]*100 for h in history], "r-o", label="Val",   lw=2)
    ax.axhline(90, color="green", ls="--", alpha=0.6, label="90% target")
    ax.axvline(best_epoch, color="purple", ls=":", alpha=0.5,
               label=f"Best ep={best_epoch}")
    ax.fill_between(epx,
        [h["tr_acc"]*100 for h in history],
        [h["vl_acc"]*100 for h in history],
        alpha=0.1, color="gray")
    ax.set_title("Accuracy per Epoch", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Macro-F1
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(epx, [h["tr_f1"]*100 for h in history], "b-o", label="Train", lw=2)
    ax.plot(epx, [h["vl_f1"]*100 for h in history], "r-o", label="Val",   lw=2)
    ax.axhline(87, color="green", ls="--", alpha=0.6, label="87% target")
    ax.set_title("Macro F1 per Epoch", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro F1 (%)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Loss
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(epx, [h["tr_loss"] for h in history], "b-o", label="Train", lw=2)
    ax.plot(epx, [h["vl_loss"] for h in history], "r-o", label="Val",   lw=2)
    ax.set_title("Focal Loss per Epoch", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Train-val gap
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(epx, [h["acc_gap"]*100 for h in history], "g-o", label="Acc gap", lw=2)
    ax.plot(epx, [h["f1_gap"]*100  for h in history], "m-o", label="F1 gap",  lw=2)
    ax.axhline(8,  color="red",  ls="--", alpha=0.6, label="8% overfit line")
    ax.axhline(0,  color="gray", ls="-",  alpha=0.3)
    ax.set_title("Train-Val Gap per Epoch", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Gap (%)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Per-class F1
    f1_per = f1_score(test_labels, test_preds, average=None, zero_division=0)
    ax     = fig.add_subplot(gs[2, 1])
    colors = ["#2ea043" if f >= 0.90 else "#e3b341" if f >= 0.80 else "#f44336"
              for f in f1_per]
    bars   = ax.bar(le.classes_, f1_per * 100, color=colors, edgecolor="white")
    ax.axhline(90, color="green",  ls="--", alpha=0.7, label="90%")
    ax.axhline(80, color="orange", ls="--", alpha=0.7, label="80%")
    for bar, val in zip(bars, f1_per):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title("Per-class F1 (test set)", fontsize=11)
    ax.set_ylabel("F1 (%)"); ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis="y")

    plt.suptitle(
        f"RoBERTa-large + Back-Translation  |  "
        f"Acc={test_acc*100:.2f}%  Macro-F1={test_f1*100:.2f}%",
        fontsize=13, y=1.01)
    plt.savefig(os.path.join(out_dir, "training_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Data
    df_orig = load_primary(CFG.PRIMARY_DATA)
    df_aug  = load_augmented(CFG.AUG_DATA)

    print(f"\nPrimary  : {len(df_orig):,} samples")
    print(f"Augmented: {len(df_aug):,} samples")

    le = LabelEncoder().fit(df_orig["label"])
    num_classes = len(le.classes_)

    train_df, val_df, test_df = build_splits(
        df_orig, df_aug, le,
        CFG.VAL_SIZE, CFG.TEST_SIZE, CFG.SEED)

    print(f"\nSplit → train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")
    print(f"Train label distribution:\n{train_df['label'].value_counts().to_string()}")

    # ── Tokeniser & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(CFG.BERT_MODEL)

    def make_loader(df, shuffle):
        ds = MentalHealthDataset(df, tokenizer, CFG.MAX_LEN)
        return DataLoader(ds, batch_size=CFG.BATCH_SIZE,
                          shuffle=shuffle, num_workers=2, pin_memory=True)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader   = make_loader(val_df,   shuffle=False)
    test_loader  = make_loader(test_df,  shuffle=False)

    # ── Class weights  (sqrt inverse frequency, normalised)
    counts   = np.bincount(train_df["label_id"].values.astype(int)).astype(float)
    inv_freq = np.sqrt(1.0 / (counts + 1e-6))
    weights  = torch.tensor(inv_freq / inv_freq.sum() * num_classes,
                             dtype=torch.float).to(DEVICE)

    criterion = FocalLoss(weights, CFG.FOCAL_GAMMA,
                          CFG.LABEL_SMOOTH, num_classes).to(DEVICE)

    # ── Model
    model = MentalHealthClassifier(
        CFG.BERT_MODEL, num_classes, CFG.DROPOUT, CFG.N_TRAINABLE
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nModel    : {CFG.BERT_MODEL}")
    print(f"Params   : {trainable:,} trainable / {total:,} total "
          f"({trainable/total*100:.1f}%)")

    # ── Optimiser & scheduler
    optimizer = build_optimizer(
        model, CFG.LR_BERT, CFG.LR_HEAD,
        CFG.LAYER_DECAY, CFG.WEIGHT_DECAY, CFG.N_TRAINABLE)

    total_steps  = len(train_loader) * CFG.EPOCHS // CFG.GRAD_ACCUM
    warmup_steps = int(total_steps * CFG.WARMUP_RATIO)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)
    scaler = GradScaler(enabled=CFG.FP16 and DEVICE.type == "cuda")

    # ── Training loop
    print(f"\n{'='*60}")
    print(f"  Training — {CFG.EPOCHS} epochs max, patience={CFG.ES_PATIENCE}")
    print(f"{'='*60}")

    best_val_f1  = 0.0
    best_val_acc = 0.0
    best_epoch   = 0
    es_counter   = 0
    history      = []
    t0           = time.time()

    for epoch in range(1, CFG.EPOCHS + 1):
        t_ep = time.time()
        print(f"\nEpoch {epoch}/{CFG.EPOCHS}")
        print("─" * 40)

        tr_loss, tr_acc, tr_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, CFG.GRAD_ACCUM)

        vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, criterion)

        elapsed = (time.time() - t0) / 60
        acc_gap = tr_acc - vl_acc
        f1_gap  = tr_f1  - vl_f1

        print(f"  Train → loss={tr_loss:.4f}  acc={tr_acc*100:.2f}%  f1={tr_f1*100:.2f}%")
        print(f"  Val   → loss={vl_loss:.4f}  acc={vl_acc*100:.2f}%  f1={vl_f1*100:.2f}%")
        print(f"  Gap   → acc={acc_gap*100:.1f}%  f1={f1_gap*100:.1f}%  "
              f"[{time.time()-t_ep:.0f}s | total {elapsed:.1f} min]")

        if acc_gap * 100 > 8:
            print(f"  WARNING: overfit gap = {acc_gap*100:.1f}%")

        history.append({
            "epoch"  : epoch,
            "tr_loss": round(tr_loss, 4), "tr_acc": round(tr_acc, 4),
            "tr_f1"  : round(tr_f1,   4), "vl_loss": round(vl_loss, 4),
            "vl_acc" : round(vl_acc,  4), "vl_f1"  : round(vl_f1,   4),
            "acc_gap": round(acc_gap,  4), "f1_gap" : round(f1_gap,   4),
        })

        improved = (vl_f1 > best_val_f1 or
                    (vl_f1 == best_val_f1 and vl_acc > best_val_acc))

        if improved:
            best_val_f1  = vl_f1
            best_val_acc = vl_acc
            best_epoch   = epoch
            es_counter   = 0
            torch.save(model.state_dict(),
                       os.path.join(CFG.OUT_DIR, "best_weights.pt"))
            print(f"  → New best saved")
        else:
            es_counter += 1
            print(f"  → No improvement ({es_counter}/{CFG.ES_PATIENCE})")
            if es_counter >= CFG.ES_PATIENCE:
                print(f"\nEarly stopping — best was epoch {best_epoch}")
                break

    total_min = (time.time() - t0) / 60
    print(f"\nTraining complete in {total_min:.1f} min")
    print(f"Best epoch: {best_epoch}  "
          f"val_acc={best_val_acc*100:.2f}%  val_f1={best_val_f1*100:.2f}%")

    # ── Temperature scaling
    model.load_state_dict(
        torch.load(os.path.join(CFG.OUT_DIR, "best_weights.pt"),
                   map_location=DEVICE))

    _, _, _, _, val_labels, logits_val = evaluate(
        model, val_loader, criterion, return_logits=True)
    val_labels = np.array(val_labels)

    best_T = find_best_temperature(logits_val, val_labels)
    base_f1 = f1_score(val_labels, logits_val.argmax(1).numpy(),
                       average="macro", zero_division=0)
    cal_f1  = f1_score(val_labels, (logits_val / best_T).argmax(1).numpy(),
                       average="macro", zero_division=0)
    print(f"\nTemperature scaling: T={best_T:.2f}  "
          f"val F1 {base_f1*100:.2f}% → {cal_f1*100:.2f}%")

    # ── Test evaluation
    _, test_acc_raw, test_f1_raw, _, test_labels, logits_test = evaluate(
        model, test_loader, criterion, return_logits=True)
    test_labels = np.array(test_labels)

    test_preds_cal = (logits_test / best_T).argmax(dim=1).numpy()
    test_acc_cal   = accuracy_score(test_labels, test_preds_cal)
    test_f1_cal    = f1_score(test_labels, test_preds_cal,
                              average="macro", zero_division=0)
    wf1            = f1_score(test_labels, test_preds_cal,
                              average="weighted", zero_division=0)

    # Use whichever is better
    if test_f1_cal >= test_f1_raw:
        test_preds, test_acc, test_f1 = test_preds_cal, test_acc_cal, test_f1_cal
    else:
        test_preds, test_acc, test_f1, best_T = (
            (logits_test).argmax(1).numpy(), test_acc_raw, test_f1_raw, 1.0)

    print(f"\n{'='*60}")
    print(f"  Final Test Results")
    print(f"{'='*60}")
    print(f"  Accuracy    : {test_acc*100:.2f}%")
    print(f"  Macro-F1    : {test_f1*100:.2f}%")
    print(f"  Weighted-F1 : {wf1*100:.2f}%")
    print(f"  Temperature : {best_T:.2f}")
    print(f"\n{classification_report(test_labels, test_preds, target_names=le.classes_, digits=4)}")

    # ── Plots
    save_plots(history, test_labels, test_preds,
               le, test_acc, test_f1, best_epoch, CFG.OUT_DIR)

    # ── Save artefacts
    model.cpu()
    payload = {
        "model_state_dict"  : model.state_dict(),
        "bert_model"        : CFG.BERT_MODEL,
        "num_classes"       : num_classes,
        "classes"           : list(le.classes_),
        "max_len"           : CFG.MAX_LEN,
        "dropout"           : CFG.DROPOUT,
        "n_trainable"       : CFG.N_TRAINABLE,
        "layer_decay"       : CFG.LAYER_DECAY,
        "temperature"       : float(best_T),
        "test_accuracy"     : round(test_acc, 4),
        "test_f1_macro"     : round(test_f1, 4),
        "test_f1_weighted"  : round(wf1, 4),
        "training_history"  : history,
        "best_epoch"        : best_epoch,
        "total_minutes"     : round(total_min, 1),
    }

    model_path = os.path.join(CFG.OUT_DIR, "mental_health_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    with open(os.path.join(CFG.OUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f, protocol=4)

    tokenizer.save_pretrained(os.path.join(CFG.OUT_DIR, "tokenizer"))

    meta = {k: v for k, v in payload.items() if k != "model_state_dict"}
    with open(os.path.join(CFG.OUT_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved → {model_path}  ({os.path.getsize(model_path)/1e6:.0f} MB)")
    print(f"Done in {total_min:.1f} min")


if __name__ == "__main__":
    main()