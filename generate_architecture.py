"""
Generate architecture diagram for the Mental Health Text Classifier.
Produces architecture.jpg matching the model pipeline in main.py.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(14, 22))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 26)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Color palette
    C_INPUT    = "#E8F5E9"   # light green
    C_TOKEN    = "#E3F2FD"   # light blue
    C_FROZEN   = "#F3E5F5"   # light purple
    C_TRAIN    = "#FFF3E0"   # light orange
    C_POOL     = "#E0F7FA"   # light cyan
    C_HEAD     = "#FDE68A"   # light yellow
    C_OUTPUT   = "#FFEBEE"   # light red
    C_BORDER   = "#37474F"   # dark gray
    C_ARROW    = "#455A64"   # arrow gray
    C_FROZEN_D = "#9C27B0"   # purple accent
    C_TRAIN_D  = "#E65100"   # orange accent
    C_TEXT     = "#212121"   # near black

    def add_box(x, y, w, h, text, facecolor, fontsize=11, bold=False,
                edgecolor=C_BORDER, linewidth=1.5, subtext=None, subtextsize=8):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=2,
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ty = y + h / 2 + (0.12 if subtext else 0)
        ax.text(x + w / 2, ty, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=C_TEXT, zorder=3)
        if subtext:
            ax.text(x + w / 2, y + h / 2 - 0.22, subtext,
                    ha="center", va="center", fontsize=subtextsize,
                    color="#616161", style="italic", zorder=3)

    def arrow(x1, y1, x2, y2, text=None, color=C_ARROW):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=2.0,
                connectionstyle="arc3,rad=0",
            ),
            zorder=1,
        )
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.25, my, text, fontsize=8, color="#757575",
                    va="center", zorder=3)

    cx = 7.0  # center x
    bw = 6.0  # box width
    bh = 0.85 # box height

    # ══════════════════════════════════════════════════════════
    # Title
    # ══════════════════════════════════════════════════════════
    ax.text(cx, 25.5, "RoBERTa-large Mental Health Classifier — Architecture",
            ha="center", va="center", fontsize=15, fontweight="bold", color=C_TEXT)
    ax.text(cx, 25.1, "Fine-tuned for 7-class mental health text classification",
            ha="center", va="center", fontsize=10, color="#757575")

    # ══════════════════════════════════════════════════════════
    # 1. Input Text
    # ══════════════════════════════════════════════════════════
    y = 24.0
    add_box(cx - bw/2, y, bw, bh, "Input Text", C_INPUT, fontsize=13, bold=True,
            subtext="Raw text string (e.g., social media post)")
    arrow(cx, y, cx, y - 0.5)

    # ══════════════════════════════════════════════════════════
    # 2. Tokenizer
    # ══════════════════════════════════════════════════════════
    y = 22.6
    add_box(cx - bw/2, y, bw, bh + 0.15, "BPE Tokenizer (RoBERTa)", C_TOKEN,
            fontsize=12, bold=True,
            subtext="max_len=256 · padding · truncation")
    # outputs
    ow = 2.5
    y_out = 21.5
    add_box(cx - ow - 0.3, y_out, ow, 0.65, "input_ids", C_TOKEN, fontsize=10,
            subtext="[256]", subtextsize=8)
    add_box(cx + 0.3, y_out, ow, 0.65, "attention_mask", C_TOKEN, fontsize=10,
            subtext="[256]", subtextsize=8)
    arrow(cx - 0.5, y, cx - ow/2 - 0.3, y_out + 0.65)
    arrow(cx + 0.5, y, cx + ow/2 + 0.3, y_out + 0.65)

    # merge arrows down
    arrow(cx - ow/2 - 0.3, y_out, cx, y_out - 0.45)
    arrow(cx + ow/2 + 0.3, y_out, cx, y_out - 0.45)

    # ══════════════════════════════════════════════════════════
    # 3. RoBERTa-large Encoder  (big grouped box)
    # ══════════════════════════════════════════════════════════
    enc_top = 20.7
    enc_h   = 6.2
    enc_w   = bw + 1.6

    # Outer encoder box
    enc_box = FancyBboxPatch(
        (cx - enc_w/2, enc_top - enc_h), enc_w, enc_h,
        boxstyle="round,pad=0.2",
        facecolor="#FAFAFA", edgecolor=C_BORDER,
        linewidth=2.0, linestyle="-", zorder=1,
    )
    ax.add_patch(enc_box)
    ax.text(cx, enc_top - 0.15, "RoBERTa-large Encoder",
            ha="center", va="center", fontsize=13, fontweight="bold", color=C_TEXT)
    ax.text(cx, enc_top - 0.50, "355M parameters · hidden_size=1024 · 24 layers",
            ha="center", va="center", fontsize=9, color="#757575")

    # -- Embedding layer (frozen)
    ey = enc_top - 1.3
    ew = enc_w - 1.0
    add_box(cx - ew/2, ey, ew, 0.70, "Embedding Layer  (FROZEN)",
            C_FROZEN, fontsize=10, bold=True,
            edgecolor=C_FROZEN_D, linewidth=1.5,
            subtext="Token + Position embeddings · ~2M params")
    arrow(cx, ey, cx, ey - 0.45)

    # -- Frozen layers 0-13
    fy = ey - 1.3
    add_box(cx - ew/2, fy, ew, 0.95, "Transformer Layers 0–13  (FROZEN)",
            C_FROZEN, fontsize=11, bold=True,
            edgecolor=C_FROZEN_D, linewidth=1.5,
            subtext="14 layers · ~226M params · General linguistic knowledge")
    # Lock icon text
    ax.text(cx + ew/2 - 0.25, fy + 0.75, "🔒", fontsize=12, ha="center", va="center", zorder=3)
    arrow(cx, fy, cx, fy - 0.45)

    # -- Trainable layers 14-23
    ty = fy - 1.55
    add_box(cx - ew/2, ty, ew, 0.95, "Transformer Layers 14–23  (TRAINABLE)",
            C_TRAIN, fontsize=11, bold=True,
            edgecolor=C_TRAIN_D, linewidth=1.5,
            subtext="10 layers · ~126M params · Layer-wise LR decay (γ=0.9)")
    ax.text(cx + ew/2 - 0.25, ty + 0.75, "🔓", fontsize=12, ha="center", va="center", zorder=3)

    # LR annotation on the side
    lr_x = cx + ew/2 + 0.3
    ax.annotate(
        "Layer 23: lr = 2e-5\nLayer 22: lr = 1.8e-5\n       ⋮\nLayer 14: lr = 7.7e-6",
        xy=(cx + ew/2, ty + 0.45),
        xytext=(lr_x + 0.1, ty + 0.45),
        fontsize=7.5, color="#795548",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E1", edgecolor="#BCAAA4", linewidth=0.8),
        arrowprops=dict(arrowstyle="->", color="#BCAAA4", lw=1.0),
        zorder=3,
    )

    # Output of encoder
    out_y = ty - 0.15
    arrow(cx, out_y, cx, out_y - 0.55)
    ax.text(cx + 0.25, out_y - 0.30, "H ∈ ℝ²⁵⁶ˣ¹⁰²⁴", fontsize=8.5,
            color="#757575", va="center", zorder=3)

    # ══════════════════════════════════════════════════════════
    # 4. Triple Pooling
    # ══════════════════════════════════════════════════════════
    pool_y = 13.4
    pw = 2.6
    ph = 0.80
    gap = 0.35

    # Triple pooling label
    ax.text(cx, pool_y + 0.55, "Triple Pooling Strategy", ha="center",
            va="center", fontsize=12, fontweight="bold", color=C_TEXT)

    # Three boxes
    x_cls  = cx - pw - gap/2 - pw/2
    x_mean = cx - pw/2
    x_max  = cx + gap/2 + pw/2

    add_box(x_cls,  pool_y - ph/2, pw, ph, "CLS Pool", C_POOL, fontsize=10, bold=True,
            subtext="H[0] → ℝ¹⁰²⁴")
    add_box(x_mean, pool_y - ph/2, pw, ph, "Mean Pool", C_POOL, fontsize=10, bold=True,
            subtext="masked avg → ℝ¹⁰²⁴")
    add_box(x_max,  pool_y - ph/2, pw, ph, "Max Pool", C_POOL, fontsize=10, bold=True,
            subtext="masked max → ℝ¹⁰²⁴")

    # Arrows from encoder to each pool
    arrow(cx - 1.5, pool_y + 0.45, x_cls + pw/2, pool_y + ph/2)
    arrow(cx,       pool_y + 0.45, x_mean + pw/2, pool_y + ph/2)
    arrow(cx + 1.5, pool_y + 0.45, x_max + pw/2, pool_y + ph/2)

    # Concatenation
    concat_y = pool_y - 1.5
    add_box(cx - bw/2, concat_y, bw, 0.70, "Concatenate  [CLS ; Mean ; Max]",
            "#E8EAF6", fontsize=11, bold=True,
            subtext="v_pool ∈ ℝ³⁰⁷²", edgecolor="#3F51B5")

    arrow(x_cls + pw/2,  pool_y - ph/2, cx - 1.5, concat_y + 0.70)
    arrow(x_mean + pw/2, pool_y - ph/2, cx,       concat_y + 0.70)
    arrow(x_max + pw/2,  pool_y - ph/2, cx + 1.5, concat_y + 0.70)

    arrow(cx, concat_y, cx, concat_y - 0.45)

    # ══════════════════════════════════════════════════════════
    # 5. Classification Head
    # ══════════════════════════════════════════════════════════
    head_top = concat_y - 0.6
    head_h = 4.2
    head_w = bw + 0.6

    head_box = FancyBboxPatch(
        (cx - head_w/2, head_top - head_h), head_w, head_h,
        boxstyle="round,pad=0.2",
        facecolor="#FFFDE7", edgecolor="#F9A825",
        linewidth=2.0, zorder=1,
    )
    ax.add_patch(head_box)
    ax.text(cx, head_top - 0.15, "Classification Head",
            ha="center", va="center", fontsize=13, fontweight="bold", color=C_TEXT)
    ax.text(cx, head_top - 0.48, "~1.6M trainable parameters · lr=1e-4",
            ha="center", va="center", fontsize=9, color="#757575")

    hw = head_w - 1.0
    ly = head_top - 1.1
    lh = 0.60
    lsp = 0.85

    # LayerNorm
    add_box(cx - hw/2, ly, hw, lh, "LayerNorm(3072)", C_HEAD, fontsize=10, bold=True)
    arrow(cx, ly, cx, ly - 0.3)

    # Layer 1: 3072→512
    ly -= lsp
    add_box(cx - hw/2, ly, hw, lh, "Linear(3072→512) + GELU + Dropout(0.30)",
            C_HEAD, fontsize=10)
    arrow(cx, ly, cx, ly - 0.3)

    # Layer 2: 512→128
    ly -= lsp
    add_box(cx - hw/2, ly, hw, lh, "Linear(512→128) + GELU + Dropout(0.15)",
            C_HEAD, fontsize=10)
    arrow(cx, ly, cx, ly - 0.3)

    # Layer 3: 128→7
    ly -= lsp
    add_box(cx - hw/2, ly, hw, lh, "Linear(128→7)  — raw logits",
            C_HEAD, fontsize=10, bold=True)

    arrow(cx, ly, cx, ly - 0.55)

    # ══════════════════════════════════════════════════════════
    # 6. Temperature-scaled Softmax
    # ══════════════════════════════════════════════════════════
    sm_y = ly - 0.7
    add_box(cx - bw/2, sm_y, bw, 0.70,
            "Temperature-Scaled Softmax (T=1.00)",
            "#F3E5F5", fontsize=11, bold=True,
            edgecolor="#7B1FA2", subtext="softmax(logits / T)")
    arrow(cx, sm_y, cx, sm_y - 0.55)

    # ══════════════════════════════════════════════════════════
    # 7. Output — 7 classes
    # ══════════════════════════════════════════════════════════
    out_y2 = sm_y - 0.75
    out_h = 1.4
    out_w = bw + 1.6
    out_box = FancyBboxPatch(
        (cx - out_w/2, out_y2 - out_h), out_w, out_h,
        boxstyle="round,pad=0.2",
        facecolor=C_OUTPUT, edgecolor="#C62828",
        linewidth=2.0, zorder=1,
    )
    ax.add_patch(out_box)
    ax.text(cx, out_y2 - 0.22, "Output: 7-Class Probability Distribution",
            ha="center", va="center", fontsize=12, fontweight="bold", color=C_TEXT)

    classes = [
        ("Normal", "#22c55e"),
        ("Depression", "#3b82f6"),
        ("Suicidal", "#ef4444"),
        ("Anxiety", "#eab308"),
        ("Bipolar", "#a855f7"),
        ("Stress", "#f97316"),
        ("Personality\nDisorder", "#a16207"),
    ]

    total_w = out_w - 0.8
    cw = total_w / 7
    sx = cx - total_w / 2
    cy_cls = out_y2 - 0.85

    for i, (name, color) in enumerate(classes):
        xi = sx + i * cw + 0.05
        cb = FancyBboxPatch(
            (xi, cy_cls - 0.25), cw - 0.1, 0.50,
            boxstyle="round,pad=0.05",
            facecolor=color + "33",  # transparent
            edgecolor=color,
            linewidth=1.2, zorder=2,
        )
        ax.add_patch(cb)
        ax.text(xi + (cw - 0.1) / 2, cy_cls, name,
                ha="center", va="center", fontsize=6.5,
                fontweight="bold", color=color, zorder=3)

    # ══════════════════════════════════════════════════════════
    # Side annotations — Loss & Training
    # ══════════════════════════════════════════════════════════
    ann_x = 0.3
    ann_y = 8.5
    ann_w = 3.0
    ann_h = 4.0
    ann_box = FancyBboxPatch(
        (ann_x, ann_y), ann_w, ann_h,
        boxstyle="round,pad=0.25",
        facecolor="#ECEFF1", edgecolor="#78909C",
        linewidth=1.2, zorder=1,
    )
    ax.add_patch(ann_box)
    ax.text(ann_x + ann_w/2, ann_y + ann_h - 0.3, "Training Config",
            ha="center", va="center", fontsize=10, fontweight="bold", color=C_TEXT)

    config_text = (
        "Loss: Focal (γ=2.0)\n"
        "  + Class weights (√1/freq)\n"
        "  + Label smooth (ε=0.05)\n\n"
        "Optimizer: AdamW (λ=0.02)\n"
        "Schedule: Cosine + 6% warmup\n"
        "Grad accum: 4 (eff. BS=64)\n"
        "FP16 mixed precision\n"
        "Grad clip: max_norm=1.0\n"
        "Early stop: patience=5"
    )
    ax.text(ann_x + 0.25, ann_y + ann_h - 0.75, config_text,
            fontsize=7.8, va="top", color="#37474F",
            fontfamily="monospace", linespacing=1.5, zorder=3)

    # ── Right side annotation — Results
    r_x = 14 - ann_w - 0.3
    r_box = FancyBboxPatch(
        (r_x, ann_y), ann_w, ann_h,
        boxstyle="round,pad=0.25",
        facecolor="#E8F5E9", edgecolor="#4CAF50",
        linewidth=1.2, zorder=1,
    )
    ax.add_patch(r_box)
    ax.text(r_x + ann_w/2, ann_y + ann_h - 0.3, "Results (Test Set)",
            ha="center", va="center", fontsize=10, fontweight="bold", color=C_TEXT)

    results_text = (
        "Accuracy  : 95.58%\n"
        "Macro F1  : 93.48%\n"
        "Weighted F1: 95.59%\n"
        "Temperature: T=1.00\n\n"
        "Best Epoch : 19/20\n"
        "Train Time : 750.8 min\n"
        "GPU        : Tesla T4\n"
        "Trainable  : 127.6M (35.8%)"
    )
    ax.text(r_x + 0.25, ann_y + ann_h - 0.75, results_text,
            fontsize=7.8, va="top", color="#2E7D32",
            fontfamily="monospace", linespacing=1.5, zorder=3)

    # ══════════════════════════════════════════════════════════
    # Legend
    # ══════════════════════════════════════════════════════════
    legend_items = [
        mpatches.Patch(facecolor=C_FROZEN, edgecolor=C_FROZEN_D, label="Frozen (not trained)"),
        mpatches.Patch(facecolor=C_TRAIN, edgecolor=C_TRAIN_D, label="Trainable (fine-tuned)"),
        mpatches.Patch(facecolor=C_POOL, edgecolor=C_BORDER, label="Pooling operations"),
        mpatches.Patch(facecolor=C_HEAD, edgecolor="#F9A825", label="Classification head"),
    ]
    ax.legend(handles=legend_items, loc="lower center",
              ncol=4, fontsize=8.5, frameon=True,
              fancybox=True, shadow=False,
              bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    plt.savefig("architecture.jpg", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.savefig("architecture.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("Saved → architecture.jpg & architecture.png")
    plt.show()


if __name__ == "__main__":
    draw_architecture()
