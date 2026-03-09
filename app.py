"""
Mental Health Text Classifier — Streamlit UI (Full Project)
=============================================================
Multi-page Streamlit application for a fine-tuned RoBERTa-large
mental health classifier. Built as a final year project showcase.

Pages:
  1. Classify     — single-text inference with detailed results
  2. Batch        — CSV upload / multi-line batch analysis
  3. Training     — interactive training history & metrics
  4. Architecture — model architecture visual explanation
  5. About        — methodology, dataset, and project details

Run with:  streamlit run app.py
"""

import os
import io
import pickle
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import altair as alt
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig


# ═══════════════════════════════════════════════════════════════
# Model definition (must match training code in main.py)
# ═══════════════════════════════════════════════════════════════

class MentalHealthClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int,
                 dropout: float, n_trainable: int):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_config(config)
        hidden = self.bert.config.hidden_size

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
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        layers = self.bert.encoder.layer
        n_freeze = max(0, len(layers) - n_trainable)
        for layer in layers[:n_freeze]:
            for p in layer.parameters():
                p.requires_grad = False

    def _mean_pool(self, hidden, mask):
        expanded = mask.unsqueeze(-1).float()
        return (hidden * expanded).sum(1) / expanded.sum(1).clamp(min=1e-9)

    def _max_pool(self, hidden, mask):
        expanded = mask.unsqueeze(-1).float()
        return (hidden * expanded + (1.0 - expanded) * -1e4).max(1).values

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state
        cls_v = seq[:, 0, :]
        mean_v = self._mean_pool(seq, attention_mask)
        max_v = self._max_pool(seq, attention_mask)
        return self.head(torch.cat([cls_v, mean_v, max_v], dim=1))


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(OUT_DIR, "mental_health_model.pkl")
LE_PATH    = os.path.join(OUT_DIR, "label_encoder.pkl")
META_PATH  = os.path.join(OUT_DIR, "model_meta.json")
TOK_DIR    = os.path.join(OUT_DIR, "tokenizer")

CLASS_ICONS = {
    "Normal":               "🟢",
    "Depression":           "🔵",
    "Suicidal":             "🔴",
    "Anxiety":              "🟡",
    "Bipolar":              "🟣",
    "Stress":               "🟠",
    "Personality disorder": "🟤",
}

CLASS_COLORS = {
    "Normal":               "#22c55e",
    "Depression":           "#3b82f6",
    "Suicidal":             "#ef4444",
    "Anxiety":              "#eab308",
    "Bipolar":              "#a855f7",
    "Stress":               "#f97316",
    "Personality disorder": "#a16207",
}

CLASS_DESCRIPTIONS = {
    "Normal": "The text does not indicate significant mental health concerns.",
    "Depression": "The text suggests signs consistent with depressive symptoms such as persistent sadness, hopelessness, or loss of interest.",
    "Suicidal": "The text contains language that may indicate suicidal ideation or self-harm thoughts. **Please take this seriously.**",
    "Anxiety": "The text suggests signs consistent with anxiety symptoms such as excessive worry, nervousness, or restlessness.",
    "Bipolar": "The text suggests signs consistent with bipolar disorder such as mood swings, manic episodes, or extreme emotional shifts.",
    "Stress": "The text suggests signs of significant stress or being overwhelmed by circumstances.",
    "Personality disorder": "The text suggests signs consistent with personality disorder traits such as unstable relationships, identity issues, or emotional dysregulation.",
}

CRISIS_RESOURCES = """
### 🆘 Crisis Support Resources

If you or someone you know is in crisis, please reach out:

| Service | Contact |
|---------|---------|
| **National Suicide Prevention Lifeline (US)** | 📞 988 or 1-800-273-8255 |
| **Crisis Text Line (US)** | 💬 Text HOME to 741741 |
| **Vandrevala Foundation (India)** | 📞 1860-2662-345 |
| **iCall (India)** | 📞 9152987821 |
| **AASRA (India)** | 📞 91-22-27546669 |
| **Befrienders Worldwide** | 🌐 [befrienders.org](https://www.befrienders.org) |

> **You are not alone.** Reaching out is a sign of strength.
"""


# ═══════════════════════════════════════════════════════════════
# Loading helpers (cached)
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model — this may take a moment on first run...")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MentalHealthClassifier(
        model_name=payload["bert_model"],
        num_classes=payload["num_classes"],
        dropout=payload["dropout"],
        n_trainable=payload["n_trainable"],
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)

    temperature = payload.get("temperature", 1.0)
    max_len = payload.get("max_len", 256)
    classes = payload.get("classes", meta.get("classes", []))

    return model, tokenizer, meta, device, temperature, max_len, classes


# ═══════════════════════════════════════════════════════════════
# Inference helpers
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_single(text: str, model, tokenizer, device,
                   temperature: float, max_len: int, classes: list):
    enc = tokenizer(
        text, max_length=max_len, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    logits = model(ids, mask)
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    confidence = float(probs[pred_idx])
    prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    return pred_label, confidence, prob_dict


@torch.no_grad()
def predict_batch(texts: list, model, tokenizer, device,
                  temperature: float, max_len: int, classes: list,
                  batch_size: int = 16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts, max_length=max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        logits = model(ids, mask)
        scaled = logits / temperature
        probs = F.softmax(scaled, dim=-1).cpu().numpy()

        for j, text in enumerate(batch_texts):
            pred_idx = int(np.argmax(probs[j]))
            results.append({
                "text": text,
                "prediction": classes[pred_idx],
                "confidence": float(probs[j][pred_idx]),
                **{f"prob_{cls}": float(probs[j][k]) for k, cls in enumerate(classes)},
            })
    return results


# ═══════════════════════════════════════════════════════════════
# Page: Classify
# ═══════════════════════════════════════════════════════════════

def page_classify(model, tokenizer, meta, device, temperature, max_len, classes):
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0;">
            <h1 style="margin-bottom:0;">🧠 Mental Health Text Classifier</h1>
            <p style="color:gray;">Analyze text for mental health indicators using fine-tuned RoBERTa-large</p>
        </div>
        """, unsafe_allow_html=True,
    )

    # Example texts — carefully chosen to match each category distinctly
    examples = {
        "— Select an example —": "",
        "Normal conversation": "I had a great day at work today. Met some friends for coffee and we talked about our weekend plans. Life feels pretty good right now.",
        "Depression indicators": "I have been sleeping all day and have no motivation to do anything. I lost interest in all my hobbies and nothing brings me joy anymore. I feel so tired and numb all the time.",
        "Anxiety indicators": "I feel like something bad is about to happen, and I can’t calm my mind down even though I know nothing is wrong.",
        "Stress indicators": "Work has been piling up and I have so many deadlines to meet. I feel mentally exhausted and burned out. The pressure from my boss and responsibilities is getting to me.",
        "Bipolar indicators": "Constant tremor I guess it is pretty common with bipolar disorder cause my buddy has the same problem. And when I experience hypomania my hands are shaking too bad especially. But is there any way out? Been kinda tired of this recently.",
        "Suicidal indicators": "I feel like everyone would be better off without me. I just want the pain to stop and I don't know how much longer I can hold on.",
    }

    selected_example = st.selectbox("Try an example:", list(examples.keys()))
    prefill = examples[selected_example] if selected_example != "— Select an example —" else ""

    text_input = st.text_area(
        "Enter text to analyze:",
        value=prefill,
        height=160,
        placeholder="Type or paste text here...",
    )

    col1, col2 = st.columns([1, 1])
    classify_clicked = col1.button("🔍 Classify", type="primary", use_container_width=True)
    clear_clicked = col2.button("🗑️ Clear", use_container_width=True)

    if clear_clicked:
        st.rerun()

    if classify_clicked:
        if not text_input or text_input.strip() == "":
            st.warning("Please enter some text to classify.")
            return

        start_time = time.time()
        with st.spinner("Analyzing..."):
            pred_label, confidence, prob_dict = predict_single(
                text_input.strip(), model, tokenizer, device,
                temperature, max_len, classes,
            )
        inference_time = time.time() - start_time

        st.markdown("---")

        # ── Prediction result
        icon = CLASS_ICONS.get(pred_label, "⬜")
        color = CLASS_COLORS.get(pred_label, "#6b7280")
        description = CLASS_DESCRIPTIONS.get(pred_label, "")

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {color}22, {color}11);
                border-left: 5px solid {color};
                border-radius: 10px;
                padding: 1.3rem 1.5rem;
                margin-bottom: 1rem;
            ">
                <p style="margin:0; font-size:0.85rem; color:gray;">Prediction</p>
                <h2 style="margin:0.2rem 0; color:{color};">{icon} {pred_label}</h2>
                <p style="margin:0; font-size:1.1rem;">
                    Confidence: <strong>{confidence * 100:.1f}%</strong>
                </p>
                <p style="margin:0.5rem 0 0 0; font-size:0.9rem; color:#888;">
                    {description}
                </p>
                <p style="margin:0.3rem 0 0 0; font-size:0.75rem; color:#aaa;">
                    Inference time: {inference_time*1000:.0f} ms
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Crisis alert
        if pred_label == "Suicidal" and confidence > 0.3:
            st.error("⚠️ **This text may indicate suicidal ideation. If this is about you or someone you know, please seek help immediately.**")
            with st.expander("🆘 Crisis Resources — Click to expand", expanded=True):
                st.markdown(CRISIS_RESOURCES)

        # ── Confidence chart
        st.markdown("#### 📊 Confidence Breakdown")

        chart_data = pd.DataFrame({
            "Category": list(prob_dict.keys()),
            "Confidence": [v * 100 for v in prob_dict.values()],
        })
        chart_data["color"] = chart_data["Category"].map(CLASS_COLORS)
        chart_data = chart_data.sort_values("Confidence", ascending=True)

        bars = (
            alt.Chart(chart_data)
            .mark_bar(cornerRadiusEnd=6)
            .encode(
                x=alt.X("Confidence:Q", title="Confidence (%)",
                         scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Category:N",
                         sort=alt.SortField("Confidence", order="ascending"),
                         title=""),
                color=alt.Color("color:N", scale=None, legend=None),
                tooltip=[
                    alt.Tooltip("Category:N"),
                    alt.Tooltip("Confidence:Q", format=".1f", title="%"),
                ],
            )
            .properties(height=260)
        )
        text_labels = bars.mark_text(
            align="left", baseline="middle", dx=5, fontSize=12
        ).encode(text=alt.Text("Confidence:Q", format=".1f"))

        st.altair_chart(bars + text_labels, use_container_width=True)

        # ── Confidence interpretation
        if confidence >= 0.85:
            conf_level = "🟢 **High confidence** — the model is very certain about this classification."
        elif confidence >= 0.60:
            conf_level = "🟡 **Moderate confidence** — the prediction is likely correct but consider the secondary categories."
        else:
            conf_level = "🔴 **Low confidence** — the model is uncertain. The text may contain mixed signals across categories."

        st.info(conf_level)

        # ── Raw probabilities
        with st.expander("📋 Detailed Probabilities"):
            prob_df = pd.DataFrame({
                "Category": list(prob_dict.keys()),
                "Probability": list(prob_dict.values()),
                "Percentage": [f"{v*100:.2f}%" for v in prob_dict.values()],
            }).sort_values("Probability", ascending=False)
            prob_df.index = range(1, len(prob_df) + 1)
            prob_df.index.name = "Rank"
            st.dataframe(
                prob_df[["Category", "Percentage"]],
                use_container_width=True,
            )

        # ── Token count info
        tokens = tokenizer.encode(text_input.strip())
        st.caption(f"📝 Input: {len(text_input.strip())} chars · {len(tokens)} tokens (max {max_len})")


# ═══════════════════════════════════════════════════════════════
# Page: Batch Analysis
# ═══════════════════════════════════════════════════════════════

def page_batch(model, tokenizer, meta, device, temperature, max_len, classes):
    st.markdown("## 📋 Batch Analysis")
    st.markdown("Classify multiple texts at once — upload a CSV or enter texts line by line.")

    tab_csv, tab_lines = st.tabs(["📁 Upload CSV", "📝 Paste Text Lines"])

    texts = []

    with tab_csv:
        st.markdown("Upload a CSV file with a column named **`text`** (or similar: `statement`, `content`, `post`, `message`).")
        uploaded = st.file_uploader("Choose CSV", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            text_col = None
            for c in df_up.columns:
                if c.lower() in ("text", "statement", "content", "post", "tweet", "message"):
                    text_col = c
                    break
            if text_col is None:
                st.error(f"No text column found. Available columns: {list(df_up.columns)}")
                return
            texts = df_up[text_col].dropna().astype(str).tolist()
            st.success(f"Found **{len(texts)}** texts in column `{text_col}`.")

    with tab_lines:
        multi_text = st.text_area(
            "Enter one text per line:",
            height=200,
            placeholder="I feel great today\nI can't stop worrying about everything\nNothing matters anymore",
        )
        if multi_text.strip():
            texts = [line.strip() for line in multi_text.strip().split("\n") if line.strip()]
            st.info(f"Found **{len(texts)}** text(s).")

    if not texts:
        return

    if len(texts) > 500:
        st.warning("Maximum 500 texts at once. Only the first 500 will be processed.")
        texts = texts[:500]

    if st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
        progress = st.progress(0, text="Classifying...")
        start_time = time.time()

        results = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = predict_batch(
                batch, model, tokenizer, device,
                temperature, max_len, classes, batch_size,
            )
            results.extend(batch_results)
            progress.progress(min(1.0, (i + batch_size) / len(texts)),
                              text=f"Processed {min(i + batch_size, len(texts))}/{len(texts)}")

        elapsed = time.time() - start_time
        progress.empty()

        st.success(f"Classified **{len(results)}** texts in **{elapsed:.1f}s** ({len(results)/max(elapsed, 0.01):.0f} texts/sec)")

        results_df = pd.DataFrame(results)

        # ── Summary metrics
        st.markdown("### 📊 Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Texts", len(results_df))
        col2.metric("Avg Confidence", f"{results_df['confidence'].mean()*100:.1f}%")
        col3.metric("Processing Time", f"{elapsed:.1f}s")

        # ── Distribution chart
        st.markdown("### Distribution of Predictions")
        dist = results_df["prediction"].value_counts().reset_index()
        dist.columns = ["Category", "Count"]
        dist["Percentage"] = (dist["Count"] / dist["Count"].sum() * 100).round(1)
        dist["color"] = dist["Category"].map(CLASS_COLORS)

        donut = (
            alt.Chart(dist)
            .mark_arc(innerRadius=60, cornerRadius=4)
            .encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color("color:N", scale=None, legend=None),
                tooltip=[
                    alt.Tooltip("Category:N"),
                    alt.Tooltip("Count:Q"),
                    alt.Tooltip("Percentage:Q", format=".1f", title="%"),
                ],
            )
            .properties(width=350, height=350)
        )

        bar_dist = (
            alt.Chart(dist)
            .mark_bar(cornerRadiusEnd=6)
            .encode(
                x=alt.X("Count:Q", title="Count"),
                y=alt.Y("Category:N", sort="-x", title=""),
                color=alt.Color("color:N", scale=None, legend=None),
                tooltip=[
                    alt.Tooltip("Category:N"),
                    alt.Tooltip("Count:Q"),
                    alt.Tooltip("Percentage:Q", format=".1f", title="%"),
                ],
            )
            .properties(height=280)
        )

        col_chart, col_bar = st.columns([1, 1])
        with col_chart:
            st.altair_chart(donut, use_container_width=True)
        with col_bar:
            st.altair_chart(bar_dist, use_container_width=True)

        # ── Suicidal alert
        suicidal_count = len(results_df[results_df["prediction"] == "Suicidal"])
        if suicidal_count > 0:
            st.error(f"⚠️ **{suicidal_count} text(s) classified as Suicidal.** Review these carefully.")
            with st.expander("🆘 Crisis Resources"):
                st.markdown(CRISIS_RESOURCES)

        # ── Confidence distribution
        st.markdown("### Confidence Distribution")
        conf_hist = (
            alt.Chart(results_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, opacity=0.8)
            .encode(
                x=alt.X("confidence:Q", bin=alt.Bin(maxbins=20),
                         title="Confidence"),
                y=alt.Y("count()", title="Number of Texts"),
                color=alt.value("#3b82f6"),
                tooltip=[alt.Tooltip("count()", title="Count")],
            )
            .properties(height=250)
        )
        st.altair_chart(conf_hist, use_container_width=True)

        # ── Results table
        st.markdown("### 📋 Full Results")
        display_df = results_df[["text", "prediction", "confidence"]].copy()
        display_df["confidence"] = display_df["confidence"].map(lambda x: f"{x*100:.1f}%")
        display_df.columns = ["Text", "Prediction", "Confidence"]
        display_df.index = range(1, len(display_df) + 1)
        st.dataframe(display_df, use_container_width=True, height=400)

        # ── Download
        csv_buf = io.StringIO()
        results_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Results CSV",
            csv_buf.getvalue(),
            file_name="classification_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════
# Page: Training Insights
# ═══════════════════════════════════════════════════════════════

def page_training(meta):
    st.markdown("## 📈 Training Insights")

    history = meta.get("training_history", [])

    if not history:
        st.warning("No training history found in model metadata.")
        return

    # ── Summary cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Epoch", meta.get("best_epoch", "?"))
    col2.metric("Test Accuracy", f"{meta.get('test_accuracy', 0)*100:.1f}%")
    col3.metric("Test Macro-F1", f"{meta.get('test_f1_macro', 0)*100:.1f}%")
    col4.metric("Training Time", f"{meta.get('total_minutes', '?')} min")

    hist_df = pd.DataFrame(history)

    # ── Accuracy plot
    st.markdown("### Accuracy over Epochs")
    acc_data = pd.melt(
        hist_df[["epoch", "tr_acc", "vl_acc"]],
        id_vars="epoch",
        value_vars=["tr_acc", "vl_acc"],
        var_name="Split",
        value_name="Accuracy",
    )
    acc_data["Accuracy"] = acc_data["Accuracy"] * 100
    acc_data["Split"] = acc_data["Split"].map({"tr_acc": "Train", "vl_acc": "Validation"})

    acc_chart = (
        alt.Chart(acc_data)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("Accuracy:Q", title="Accuracy (%)", scale=alt.Scale(zero=False)),
            color=alt.Color("Split:N", scale=alt.Scale(
                domain=["Train", "Validation"],
                range=["#3b82f6", "#ef4444"]),
            ),
            tooltip=[
                alt.Tooltip("epoch:Q", title="Epoch"),
                alt.Tooltip("Split:N"),
                alt.Tooltip("Accuracy:Q", format=".2f", title="Accuracy (%)"),
            ],
        )
        .properties(height=350)
    )

    target_line = (
        alt.Chart(pd.DataFrame({"y": [90]}))
        .mark_rule(strokeDash=[5, 5], color="green", opacity=0.6)
        .encode(y="y:Q")
    )

    best_ep_line = (
        alt.Chart(pd.DataFrame({"x": [meta.get("best_epoch", 1)]}))
        .mark_rule(strokeDash=[3, 3], color="purple", opacity=0.5)
        .encode(x="x:Q")
    )

    st.altair_chart(acc_chart + target_line + best_ep_line, use_container_width=True)

    # ── F1 and Loss side by side
    col_f1, col_loss = st.columns(2)

    with col_f1:
        st.markdown("### Macro F1")
        f1_data = pd.melt(
            hist_df[["epoch", "tr_f1", "vl_f1"]],
            id_vars="epoch", value_vars=["tr_f1", "vl_f1"],
            var_name="Split", value_name="F1",
        )
        f1_data["F1"] = f1_data["F1"] * 100
        f1_data["Split"] = f1_data["Split"].map({"tr_f1": "Train", "vl_f1": "Validation"})

        f1_chart = (
            alt.Chart(f1_data)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("F1:Q", title="Macro F1 (%)", scale=alt.Scale(zero=False)),
                color=alt.Color("Split:N", scale=alt.Scale(
                    domain=["Train", "Validation"],
                    range=["#3b82f6", "#ef4444"])),
                tooltip=[
                    alt.Tooltip("epoch:Q"),
                    alt.Tooltip("Split:N"),
                    alt.Tooltip("F1:Q", format=".2f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(f1_chart, use_container_width=True)

    with col_loss:
        st.markdown("### Focal Loss")
        loss_data = pd.melt(
            hist_df[["epoch", "tr_loss", "vl_loss"]],
            id_vars="epoch", value_vars=["tr_loss", "vl_loss"],
            var_name="Split", value_name="Loss",
        )
        loss_data["Split"] = loss_data["Split"].map({"tr_loss": "Train", "vl_loss": "Validation"})

        loss_chart = (
            alt.Chart(loss_data)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("Loss:Q", title="Loss", scale=alt.Scale(zero=False)),
                color=alt.Color("Split:N", scale=alt.Scale(
                    domain=["Train", "Validation"],
                    range=["#3b82f6", "#ef4444"])),
                tooltip=[
                    alt.Tooltip("epoch:Q"),
                    alt.Tooltip("Split:N"),
                    alt.Tooltip("Loss:Q", format=".4f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(loss_chart, use_container_width=True)

    # ── Overfitting gap
    st.markdown("### Train-Validation Gap (Overfitting Monitor)")
    gap_data = pd.melt(
        hist_df[["epoch", "acc_gap", "f1_gap"]],
        id_vars="epoch", value_vars=["acc_gap", "f1_gap"],
        var_name="Metric", value_name="Gap",
    )
    gap_data["Gap"] = gap_data["Gap"] * 100
    gap_data["Metric"] = gap_data["Metric"].map({"acc_gap": "Accuracy Gap", "f1_gap": "F1 Gap"})

    gap_chart = (
        alt.Chart(gap_data)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("Gap:Q", title="Gap (%)"),
            color=alt.Color("Metric:N", scale=alt.Scale(
                domain=["Accuracy Gap", "F1 Gap"],
                range=["#22c55e", "#a855f7"])),
            tooltip=[
                alt.Tooltip("epoch:Q"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Gap:Q", format=".2f"),
            ],
        )
        .properties(height=300)
    )

    overfit_line = (
        alt.Chart(pd.DataFrame({"y": [8]}))
        .mark_rule(strokeDash=[5, 5], color="red", opacity=0.6)
        .encode(y="y:Q")
    )

    st.altair_chart(gap_chart + overfit_line, use_container_width=True)
    st.caption("The red dashed line marks the 8% overfitting threshold.")

    # ── Epoch-by-epoch table
    with st.expander("📋 Full Training Log"):
        display_hist = hist_df.copy()
        for col in ["tr_acc", "vl_acc", "tr_f1", "vl_f1", "acc_gap", "f1_gap"]:
            if col in display_hist.columns:
                display_hist[col] = (display_hist[col] * 100).round(2).astype(str) + "%"
        st.dataframe(display_hist, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# Page: Model Architecture
# ═══════════════════════════════════════════════════════════════

def page_architecture(meta):
    st.markdown("## 🏗️ Model Architecture")
    st.markdown("A visual overview of the fine-tuned RoBERTa-large classifier pipeline.")

    # ── Pipeline diagram (image)
    st.markdown("### Processing Pipeline")
    arch_img_path = os.path.join(BASE_DIR, "architecture1.png")
    if os.path.exists(arch_img_path):
        st.image(arch_img_path, caption="RoBERTa-large Mental Health Classifier Pipeline", use_container_width=True)
    else:
        st.warning(
            "Architecture diagram image not found. "
            "Please save `architecture_diagram.png` in the project root directory."
        )

    # ── Key design decisions
    st.markdown("### 🔑 Key Design Decisions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Partial Fine-tuning**
            - Bottom 14 layers frozen → preserves general language understanding
            - Top 10 layers trained → adapts to mental health domain
            - Reduces overfitting risk on ~52K samples

            **Triple Pooling**
            - CLS token: overall sentence representation
            - Mean pool: captures average semantic content
            - Max pool: captures most salient features
            - 3× richer than CLS-only (3072 vs 1024 dims)

            **Layer-wise LR Decay (γ=0.9)**
            - Top layer gets full learning rate
            - Each lower layer gets 0.9× the rate above
            - Prevents lower layers from drifting
            """
        )

    with col2:
        st.markdown(
            f"""
            **Focal Loss + Label Smoothing**
            - Focal loss (γ=2.0): down-weights easy examples
            - Class weighting: √(1/frequency) for imbalance
            - Label smoothing (ε=0.05): prevents overconfidence

            **Training Strategy**
            - Gradient accumulation: 4 steps (eff. batch=64)
            - FP16 mixed precision for memory efficiency
            - Cosine schedule with 6% warmup
            - Early stopping (patience=5)

            **Post-training Calibration**
            - Temperature scaling on validation set
            - T={meta.get('temperature', 1.0):.2f} found via grid search
            - Improves probability calibration
            """
        )

    # ── Parameter summary
    st.markdown("### 📊 Parameter Summary")
    param_data = pd.DataFrame({
        "Component": ["Embedding layer", "Frozen encoder (layers 0–13)", "Trainable encoder (layers 14–23)", "Classification head", "**Total**"],
        "Parameters": ["~2M", "~186M", "~133M", "~1.6M", "~355M"],
        "Trainable": ["❌ No", "❌ No", "✅ Yes", "✅ Yes", "~134.6M (38%)"],
    })
    st.table(param_data)


# ═══════════════════════════════════════════════════════════════
# Page: About
# ═══════════════════════════════════════════════════════════════

def page_about(meta, classes):
    st.markdown("## 📄 About This Project")

    st.markdown(
        """
        ### Overview

        This project implements an **NLP-based mental health text classifier** that
        analyzes text and identifies potential mental health indicators. It uses a
        fine-tuned **RoBERTa-large** transformer model to classify text into 7 categories.

        The system is designed as a research tool to assist — not replace — mental health
        professionals by providing automated preliminary screening of text data.

        ---

        ### Problem Statement

        Mental health conditions affect millions globally, yet many go undiagnosed.
        Social media and online platforms contain valuable textual signals that could
        enable early detection. This project explores whether modern NLP techniques
        can reliably identify mental health indicators from text.

        ---

        ### Dataset

        The model is trained on the **Sentiment Analysis for Mental Health** dataset
        from Kaggle, which contains labeled text samples across 7 categories:
        """
    )

    cat_data = []
    for cls in classes:
        icon = CLASS_ICONS.get(cls, "⬜")
        desc = CLASS_DESCRIPTIONS.get(cls, "")
        cat_data.append({"Category": f"{icon} {cls}", "Description": desc})
    st.table(pd.DataFrame(cat_data))

    st.markdown(
        f"""
        ---

        ### Methodology

        1. **Data Preprocessing**
           - Text cleaning and normalization
           - Minimum length filtering (≥8 characters)
           - Stratified train/val/test split (80/10/10)

        2. **Data Augmentation**
           - Back-translation (English → target language → English)
           - Augmented data added to training set only
           - Validation and test sets remain clean (no data leakage)

        3. **Model Training**
           - Base model: RoBERTa-large (355M parameters)
           - Partial fine-tuning: top 10 of 24 transformer layers
           - Layer-wise learning rate decay (γ=0.9)
           - Focal loss with inverse-frequency class weighting
           - Label smoothing (ε=0.05) to prevent overconfidence
           - FP16 mixed precision training
           - Cosine learning rate schedule with 6% warmup
           - Early stopping with patience of 5 epochs

        4. **Post-processing**
           - Temperature scaling for probability calibration
           - T={meta.get('temperature', 1.0):.2f} found via grid search on validation set

        ---

        ### Results

        | Metric | Score |
        |--------|-------|
        | **Test Accuracy** | {meta.get('test_accuracy', 0)*100:.2f}% |
        | **Test Macro F1** | {meta.get('test_f1_macro', 0)*100:.2f}% |
        | **Test Weighted F1** | {meta.get('test_f1_weighted', 0)*100:.2f}% |
        | **Best Epoch** | {meta.get('best_epoch', '?')} |
        | **Training Time** | {meta.get('total_minutes', '?')} minutes |

        ---

        ### Tech Stack

        | Component | Technology |
        |-----------|-----------|
        | Language | Python 3 |
        | Deep Learning | PyTorch |
        | NLP | HuggingFace Transformers |
        | Base Model | RoBERTa-large |
        | UI Framework | Streamlit |
        | Visualization | Altair |
        | Data Processing | pandas, scikit-learn |

        ---

        ### Limitations

        - The model is trained on **English text only**
        - Classification is based on textual patterns, not clinical diagnosis
        - Performance may vary on text from different domains or demographics
        - The model should **never** be used as a sole basis for clinical decisions
        - Predictions reflect statistical patterns, not individual assessment

        ---

        ### Ethical Considerations

        - This tool is for **research and educational purposes only**
        - It is **not** a substitute for professional mental health evaluation
        - Users analyzing sensitive content should follow appropriate ethical guidelines
        - Crisis resources are automatically provided when suicidal content is detected
        - No user data is stored or transmitted — all inference runs locally
        """
    )


# ═══════════════════════════════════════════════════════════════
# Sidebar (shared across all pages)
# ═══════════════════════════════════════════════════════════════

def render_sidebar(meta, classes, device):
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 0.5rem 0;">
                <h2 style="margin:0;">🧠</h2>
                <h3 style="margin:0;">Mental Health<br>Text Classifier</h3>
                <p style="color:gray; font-size:0.8rem; margin:0.2rem 0;">v1.0 · RoBERTa-large</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Theme toggle
        dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.get("dark_mode", True))
        st.session_state["dark_mode"] = dark_mode

        st.divider()

        st.markdown("### 📊 Test Performance")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{meta.get('test_accuracy', 0)*100:.1f}%")
        col2.metric("Macro F1", f"{meta.get('test_f1_macro', 0)*100:.1f}%")

        st.divider()

        st.markdown("### 🏷️ Categories")
        for cls in classes:
            icon = CLASS_ICONS.get(cls, "⬜")
            st.markdown(f"&nbsp; {icon} {cls}")

        st.divider()

        st.caption(f"⚙️ Device: `{device}`")
        st.caption(f"🌡️ Temperature: `{meta.get('temperature', 1.0):.2f}`")
        st.caption(f"📏 Max tokens: `{meta.get('max_len', 256)}`")


# ═══════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Mental Health Classifier",
        page_icon="🧠",
        layout="wide",
    )

    # ── Load model
    try:
        model, tokenizer, meta, device, temperature, max_len, classes = load_model()
    except FileNotFoundError as e:
        st.error(f"**Model files not found.** Run training (`python main.py`) first.\n\n`{e}`")
        st.stop()
        return  # safety fallback for bare-mode (python app.py) where st.stop() is a no-op
    except Exception as e:
        st.error(f"**Error loading model:** `{e}`")
        st.stop()
        return

    # ── Sidebar
    render_sidebar(meta, classes, device)

    # ── Theme CSS (applied after sidebar so toggle state is known)
    dark = st.session_state.get("dark_mode", True)

    if dark:
        theme_css = """
        <style>
        /* ── Dark theme ───────────────────────────── */
        :root {
            color-scheme: dark;
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
            color: #fafafa;
        }
        [data-testid="stSidebar"] {
            background-color: #161b22;
        }
        [data-testid="stHeader"] {
            background-color: #0e1117;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #161b22;
            border-radius: 10px;
            padding: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            font-size: 1rem;
            color: #c9d1d9;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #21262d;
            color: #58a6ff;
        }
        [data-testid="stMetric"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 12px;
        }
        [data-testid="stExpander"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
        }
        .stDataFrame, .stTable {
            border-color: #30363d;
        }
        div[data-testid="stAlert"] {
            border-radius: 10px;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #e6edf3;
        }
        .stMarkdown a {
            color: #58a6ff;
        }
        </style>
        """
    else:
        theme_css = """
        <style>
        /* ── Light theme ──────────────────────────── */
        :root {
            color-scheme: light;
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #ffffff;
            color: #24292f;
        }
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
            color: #24292f;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
            color: #24292f !important;
        }
        [data-testid="stHeader"] {
            background-color: #ffffff;
        }
        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            font-size: 1rem;
            color: #57606a;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #ffffff;
            color: #0969da;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        /* ── Inputs, selects, text areas ── */
        .stTextInput input, .stTextArea textarea, .stSelectbox select,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {
            background-color: #ffffff !important;
            color: #24292f !important;
            border-color: #d0d7de !important;
        }
        [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
            color: #24292f !important;
        }
        /* ── Dropdown menu ── */
        [data-baseweb="popover"], [data-baseweb="menu"],
        [role="listbox"], [role="option"] {
            background-color: #ffffff !important;
            color: #24292f !important;
        }
        [role="option"]:hover {
            background-color: #f0f2f6 !important;
        }
        /* ── Buttons ── */
        .stButton > button {
            background-color: #f0f2f6;
            color: #24292f;
            border: 1px solid #d0d7de;
        }
        .stButton > button:hover {
            background-color: #e2e5e9;
            border-color: #b0b8c1;
        }
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="stBaseButton-primary"] {
            background-color: #0969da;
            color: #ffffff;
            border: none;
        }
        /* ── Metrics ── */
        [data-testid="stMetric"] {
            background-color: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 10px;
            padding: 12px;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: #24292f !important;
        }
        /* ── Expanders ── */
        [data-testid="stExpander"] {
            background-color: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 10px;
        }
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] p {
            color: #24292f !important;
        }
        /* ── Tables / Dataframes ── */
        .stDataFrame, .stTable {
            border-color: #d0d7de;
        }
        .stDataFrame th, .stTable th {
            background-color: #f0f2f6 !important;
            color: #24292f !important;
        }
        .stDataFrame td, .stTable td {
            color: #24292f !important;
        }
        /* ── Toggle ── */
        [data-testid="stToggle"] label span {
            color: #24292f !important;
        }
        /* ── Alerts / Info / Warning boxes ── */
        div[data-testid="stAlert"] {
            border-radius: 10px;
        }
        /* ── File uploader ── */
        [data-testid="stFileUploader"] {
            background-color: #f6f8fa;
            border-color: #d0d7de;
        }
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] p {
            color: #24292f !important;
        }
        /* ── General ── */
        .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #24292f;
        }
        p, li, span, label, div {
            color: #24292f;
        }
        .stMarkdown a {
            color: #0969da;
        }
        /* ── Captions ── */
        .stCaption, [data-testid="stCaptionContainer"] {
            color: #57606a !important;
        }
        /* ── Download button ── */
        .stDownloadButton > button {
            background-color: #f0f2f6;
            color: #24292f;
            border: 1px solid #d0d7de;
        }
        </style>
        """

    st.markdown(theme_css, unsafe_allow_html=True)

    # ── Navigation tabs
    tab_classify, tab_batch, tab_training, tab_arch, tab_about = st.tabs([
        "🔍 Classify",
        "📋 Batch Analysis",
        "📈 Training Insights",
        "🏗️ Architecture",
        "📄 About",
    ])

    with tab_classify:
        page_classify(model, tokenizer, meta, device, temperature, max_len, classes)

    with tab_batch:
        page_batch(model, tokenizer, meta, device, temperature, max_len, classes)

    with tab_training:
        page_training(meta)

    with tab_arch:
        page_architecture(meta)

    with tab_about:
        page_about(meta, classes)

    # ── Footer
    # st.markdown("---")
    # st.markdown(
    #     "<p style='text-align:center; color:gray; font-size:0.85rem;'>"
    #     "⚠️ This tool is for <strong>research and educational purposes only</strong>. "
    #     "It is not a substitute for professional mental health assessment. "
    #     "If you or someone you know is in crisis, please contact a mental health professional or crisis helpline."
    #     "</p>",
    #     unsafe_allow_html=True,
    # )


if __name__ == "__main__":
    main()
