import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install",
                "transformers>=4.41.0", "sentencepiece", "--quiet", "--upgrade"], check=True)

import os, time, warnings
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer

warnings.filterwarnings("ignore")

# ================================================================
#  Back-Translation Dataset Augmenter
#  Targets: Depression, Suicidal, Personality disorder
#  Paths:   EN→FR→EN  |  EN→DE→EN  |  EN→ES→EN
# ================================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Config ──────────────────────────────────────────────────────
DATAPATH        = "./data/Combined Data.csv"
OUTDIR          = "./data"
AUG_CACHE       = "./augmented_data.csv"

TRANS_PATHS = [
    ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en", "fr"),
    ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en", "de"),
    ("Helsinki-NLP/opus-mt-en-es", "Helsinki-NLP/opus-mt-es-en", "es"),
]
AUGMENT_CLASSES  = ["Depression", "Suicidal", "Personality disorder"]
TRANS_BATCH_SIZE = 32
MAX_TRANS_LEN    = 256

LABELMAP = {
    "normal"              : "Normal",
    "depression"          : "Depression",
    "suicidal"            : "Suicidal",
    "anxiety"             : "Anxiety",
    "bipolar"             : "Bipolar",
    "stress"              : "Stress",
    "personality disorder": "Personality disorder",
}

os.makedirs(OUTDIR, exist_ok=True)


# ── Load & clean primary dataset ────────────────────────────────
print("\n" + "="*60)
print("  STEP 1 — LOAD PRIMARY DATASET")
print("="*60)

df = pd.read_csv(DATAPATH)

TEXTCOL  = next(c for c in df.columns if c.lower() in
                ["text", "statement", "post", "content", "tweet"])
LABELCOL = next(c for c in df.columns if c.lower() in
                ["label", "status", "class", "category", "target"])

df = (df[[TEXTCOL, LABELCOL]]
        .rename(columns={TEXTCOL: "text", LABELCOL: "label"}))
df["text"]  = df["text"].astype(str).str.strip()
df["label"] = df["label"].str.lower().map(LABELMAP).fillna(df["label"])
df = df[df["text"].str.len() >= 8].dropna(subset=["label"]).reset_index(drop=True)
df["source"] = "original"

print(f"  Loaded {len(df)} samples")
print(df["label"].value_counts().to_string())


# ── Translation helpers ──────────────────────────────────────────
def translate_batch(texts, model, tokenizer, device,
                    max_len=256, batch_size=32):
    """Translate a list of strings; returns translated strings."""
    results = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch   = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)
        with torch.no_grad():
            out = model.generate(**encoded, num_beams=4, max_length=max_len)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        results.extend(decoded)

        done = min(i + batch_size, len(texts))
        if (i // batch_size) % 10 == 0:
            print(f"    {done}/{len(texts)} translated …", flush=True)

    return results


def back_translate(texts, fwd_name, bwd_name, lang, device):
    """EN → <lang> → EN.  Frees VRAM between the two model loads."""
    print(f"\n  Loading forward model  ({fwd_name}) …")
    fwd_tok   = MarianTokenizer.from_pretrained(fwd_name)
    fwd_model = MarianMTModel.from_pretrained(fwd_name).to(device)

    print(f"  EN → {lang.upper()} …")
    intermediate = translate_batch(
        texts, fwd_model, fwd_tok, device, MAX_TRANS_LEN, TRANS_BATCH_SIZE)

    del fwd_model
    torch.cuda.empty_cache()

    print(f"\n  Loading backward model ({bwd_name}) …")
    bwd_tok   = MarianTokenizer.from_pretrained(bwd_name)
    bwd_model = MarianMTModel.from_pretrained(bwd_name).to(device)

    print(f"  {lang.upper()} → EN …")
    back = translate_batch(
        intermediate, bwd_model, bwd_tok, device, MAX_TRANS_LEN, TRANS_BATCH_SIZE)

    del bwd_model
    torch.cuda.empty_cache()

    return back


# ── Main augmentation loop ───────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2 — BACK-TRANSLATION")
print("="*60)

if os.path.exists(AUG_CACHE):
    # ── Fast path: cache hit ─────────────────────────────────────
    print(f"\n  Cache found → loading {AUG_CACHE}")
    df_aug = pd.read_csv(AUG_CACHE)
    print(f"  {len(df_aug)} augmented samples loaded")
    print(df_aug["label"].value_counts().to_string())

else:
    # ── Slow path: run translations ──────────────────────────────
    print("\n  No cache — running back-translation (~45–60 min on GPU)")

    df_to_aug = df[df["label"].isin(AUGMENT_CLASSES)].copy()
    print(f"\n  Samples selected for augmentation : {len(df_to_aug)}")
    print(df_to_aug["label"].value_counts().to_string())

    texts  = df_to_aug["text"].tolist()
    labels = df_to_aug["label"].tolist()
    original_set = set(df_to_aug["text"].str.lower().str.strip())

    aug_frames = []
    t0 = time.time()

    for fwd_name, bwd_name, lang in TRANS_PATHS:
        print(f"\n  {'─'*50}")
        print(f"  PATH: EN → {lang.upper()} → EN")
        print(f"  {'─'*50}")
        try:
            back_texts = back_translate(texts, fwd_name, bwd_name, lang, DEVICE)

            df_lang = pd.DataFrame({
                "text"  : back_texts,
                "label" : labels,
                "source": f"backtrans_{lang}",
            })

            # Drop rows identical to originals (translation stalled)
            before = len(df_lang)
            df_lang = df_lang[
                ~df_lang["text"].str.lower().str.strip().isin(original_set)
            ]
            df_lang = df_lang[df_lang["text"].str.len() >= 8]

            removed = before - len(df_lang)
            print(f"\n  ✅  {lang.upper()}: kept {len(df_lang)} "
                  f"(removed {removed} identical/empty)")
            aug_frames.append(df_lang)

        except Exception as exc:
            print(f"\n  ⚠️  {lang.upper()} failed: {exc}")
            print("      Skipping this path and continuing …")
            torch.cuda.empty_cache()

    elapsed = (time.time() - t0) / 60
    print(f"\n  Back-translation finished in {elapsed:.1f} min")

    if aug_frames:
        df_aug = pd.concat(aug_frames, ignore_index=True)
        df_aug = df_aug.drop_duplicates(subset=["text"])

        df_aug.to_csv(AUG_CACHE, index=False)
        print(f"\n  💾  Saved to {AUG_CACHE}")
        print(f"  Total new samples : {len(df_aug)}")
        print(df_aug["label"].value_counts().to_string())
    else:
        print("\n  ⚠️  All paths failed — df_aug is empty")
        df_aug = pd.DataFrame(columns=["text", "label", "source"])


# ── Merge & save final combined CSV ─────────────────────────────
print("\n" + "="*60)
print("  STEP 3 — MERGE & SAVE")
print("="*60)

if len(df_aug) > 0:
    df_final = (pd.concat([df, df_aug], ignore_index=True)
                  .query("text.str.len() >= 8")
                  .reset_index(drop=True))
else:
    df_final = df.copy()

out_path = os.path.join(OUTDIR, "combined_augmented.csv")
df_final.to_csv(out_path, index=False)

print(f"\n  Original  : {len(df):>7,}")
print(f"  Augmented : {len(df_aug):>7,}")
print(f"  Total     : {len(df_final):>7,}")
print(f"\n  Label distribution (final):")
print(df_final["label"].value_counts().to_string())
print(f"\n  ✅  Saved → {out_path}")