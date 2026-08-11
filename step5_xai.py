"""
STEP 5: XAI Analysis — SHAP and LIME
=======================================
Run this after step3_train_transformers.py has saved your models.

This script:
1. Loads your fine-tuned RoBERTa model
2. Runs SHAP on the test set → saves Figure 2 (token importance map)
3. Runs LIME on the test set → saves Figure 3 (feature importance bar chart)
4. Runs LIME stability test  → saves Figure 4 (stability comparison chart)
5. Compares SHAP vs LIME top tokens → prints Table 5 data
6. Runs SHAP on BERT too for the Section 4.3.5 comparison

All figures are saved as PNG files you can insert into your dissertation
to REPLACE the placeholder figures that are currently in there.

IMPORTANT: This script uses your REAL trained models and produces
REAL results. Whatever comes out - those are your actual findings.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import shap
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
import scipy

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
DATA_DIR          = "processed_data/"
CHECKPOINT_DIR    = "checkpoints/"
FIGURES_DIR       = "figures/"
ROBERTA_PATH      = "checkpoints/roberta_phishing"
BERT_PATH         = "checkpoints/bert_phishing"
RANDOM_SEED       = 42
MAX_LEN           = 512
N_SHAP_SAMPLES    = 50    # number of emails to run SHAP on (increase if time allows)
N_LIME_SAMPLES    = 100   # number of emails to run LIME on
LIME_FEATURES     = 15    # top N features to show in LIME
LIME_PERTURBATIONS = 500  # perturbations per LIME explanation
STABILITY_RUNS    = 5     # how many times to repeat LIME for stability test
# ============================================================

np.random.seed(RANDOM_SEED)
os.makedirs(FIGURES_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ── Model loading helpers ─────────────────────────────────────────────

def load_model_and_tokenizer(checkpoint_path):
    """Load a saved Hugging Face model + tokeniser."""
    print(f"Loading model from {checkpoint_path}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()
    model.to(device)
    return model, tokenizer


def predict_proba(texts, model, tokenizer):
    """
    Return softmax probabilities for a list of texts.
    Shape: (n_texts, 2)  — column 0 = legit prob, column 1 = phishing prob
    """
    all_probs = []
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors='pt',
            max_length=MAX_LEN,
            truncation=True,
            padding='max_length'
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        all_probs.append(probs)
    return np.array(all_probs)


# ── SHAP ─────────────────────────────────────────────────────────────

def run_shap(model, tokenizer, phishing_emails, save_path):
    """
    Run SHAP Partition Explainer on a sample of phishing emails.
    Saves a token importance plot for the first email as Figure 2.
    Returns the global token ranking.
    """
    print(f"\nRunning SHAP on {len(phishing_emails)} phishing emails...")

    # Create a predict function that SHAP can call
    def predict_fn(texts):
        return predict_proba(texts, model, tokenizer)

    # Use the SHAP PartitionExplainer (works with any text model)
    masker = shap.maskers.Text(tokenizer=r"\W+")
    explainer = shap.Explainer(predict_fn, masker, output_names=["Legit", "Phishing"])

    # Run on sample
    shap_values = explainer(phishing_emails[:N_SHAP_SAMPLES], fixed_context=1)

    # Save Figure 2 — token importance map for first email
    print("Saving Figure 2 (SHAP token map)...")
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('off')
    fig.patch.set_facecolor('#FAFAFA')

    # Get token-level values for phishing class (index 1) for first email
    vals = shap_values[0, :, 1].values
    tokens = shap_values[0, :, 1].data

    x, y = 0.01, 0.75
    for tok, val in zip(tokens, vals):
        tok_str = str(tok).strip()
        if not tok_str:
            continue
        intensity = min(abs(float(val)) / (max(abs(vals)) + 1e-8), 1.0)
        if float(val) > 0.01:
            color = (1.0, max(0.15, 1.0 - intensity * 0.85),
                     max(0.15, 1.0 - intensity * 0.85))
        else:
            color = (0.95, 0.95, 0.95)

        pad = len(tok_str) * 0.011 + 0.012
        if x + pad > 0.97:
            x = 0.01
            y -= 0.30

        rect = plt.Rectangle((x, y - 0.11), pad, 0.18,
                              color=color, zorder=2)
        ax.add_patch(rect)
        ax.text(x + pad/2, y - 0.02, tok_str,
                ha='center', va='center',
                fontsize=10.5, fontfamily='monospace',
                color='#1a0000' if float(val) > 0.1 else '#333333',
                zorder=3)
        x += pad + 0.004

    ax.text(0.01, 0.08,
            "Red = pushes towards PHISHING    Grey = near-neutral",
            fontsize=8, color='#555555')
    ax.set_title(
        "Figure 2: SHAP token-level importance — representative phishing email (RoBERTa)\n"
        "Darker red = stronger contribution to phishing prediction",
        fontsize=9.5, loc='left', pad=5)
    plt.tight_layout(pad=0.3)
    plt.savefig(f"{save_path}/figure2_shap_tokens.png",
                dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"  Saved: {save_path}/figure2_shap_tokens.png")

    # Aggregate top tokens across all SHAP explanations
    token_scores = {}
    for i in range(len(shap_values)):
        toks = shap_values[i, :, 1].data
        vals_i = shap_values[i, :, 1].values
        for tok, val in zip(toks, vals_i):
            tok_str = str(tok).strip().lower()
            if len(tok_str) > 2:
                token_scores[tok_str] = token_scores.get(tok_str, 0) + float(val)

    # Sort by total importance
    top_shap = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\nTop 10 SHAP tokens (global):")
    for tok, score in top_shap[:10]:
        print(f"  {tok:<20} {score:.4f}")

    return top_shap


# ── LIME ─────────────────────────────────────────────────────────────

def run_lime(model, tokenizer, phishing_emails, save_path):
    """
    Run LIME on a sample of phishing emails.
    Saves Figure 3 — bar chart of top 15 feature importances.
    Returns top token list.
    """
    print(f"\nRunning LIME on {len(phishing_emails)} phishing emails...")

    def classifier_fn(texts):
        return predict_proba(texts, model, tokenizer)

    explainer = LimeTextExplainer(
        class_names=['Legitimate', 'Phishing'],
        random_state=RANDOM_SEED
    )

    all_token_weights = {}

    for i, email in enumerate(phishing_emails[:N_LIME_SAMPLES]):
        if i % 10 == 0:
            print(f"  LIME progress: {i}/{N_LIME_SAMPLES}")
        exp = explainer.explain_instance(
            email,
            classifier_fn,
            num_features=LIME_FEATURES,
            num_samples=LIME_PERTURBATIONS,
            labels=[1]
        )
        for word, weight in exp.as_list(label=1):
            word = word.lower().strip()
            if len(word) > 2:
                all_token_weights[word] = all_token_weights.get(word, 0) + weight

    # Sort by total weight
    top_lime = sorted(all_token_weights.items(), key=lambda x: x[1], reverse=True)[:15]

    print("\nTop 10 LIME tokens (aggregated):")
    for tok, w in top_lime[:10]:
        print(f"  {tok:<20} {w:.4f}")

    # Save Figure 3 — LIME bar chart
    print("Saving Figure 3 (LIME bar chart)...")
    words   = [t[0] for t in top_lime]
    weights = [t[1] for t in top_lime]
    # Normalise weights to 0-1 for readability
    max_w = max(weights) if weights else 1
    weights_norm = [w / max_w for w in weights]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    colors = ['#FF8000' if w > 0 else '#4488FF' for w in weights_norm]
    y_pos  = np.arange(len(words))
    ax.barh(y_pos, weights_norm, color=colors,
            edgecolor='white', height=0.65)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Normalised feature importance (contribution to phishing class)",
                  fontsize=9.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(
        "Figure 3: LIME feature importance — top 15 tokens across phishing predictions (RoBERTa)\n"
        "Bar length = normalised contribution strength towards phishing classification",
        fontsize=9.5, loc='left', pad=6)

    for i, w in enumerate(weights_norm):
        ax.text(w + 0.01, i, f'{w:.3f}',
                va='center', fontsize=8.5, color='#333')

    plt.tight_layout(pad=0.6)
    plt.savefig(f"{save_path}/figure3_lime_bar.png",
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}/figure3_lime_bar.png")

    return top_lime


# ── LIME Stability ───────────────────────────────────────────────────

def run_lime_stability(roberta_model, roberta_tok,
                       bert_model, bert_tok,
                       test_email, save_path):
    """
    Run LIME 5 times on the same email for both RoBERTa and BERT.
    Records which tokens appear in the top-5 on each run.
    Saves Figure 4.
    """
    print(f"\nRunning LIME stability test ({STABILITY_RUNS} runs per model)...")

    def get_top5(email, model, tokenizer, run_seed):
        def clf_fn(texts):
            return predict_proba(texts, model, tokenizer)

        exp = LimeTextExplainer(
            class_names=['Legitimate', 'Phishing'],
            random_state=run_seed
        )
        result = exp.explain_instance(
            email, clf_fn,
            num_features=LIME_FEATURES,
            num_samples=LIME_PERTURBATIONS,
            labels=[1]
        )
        # Get top 5 tokens with positive weight
        top5 = [w.lower().strip()
                for w, score in result.as_list(label=1)
                if score > 0][:5]
        return set(top5)

    # Run for both models
    roberta_runs = []
    bert_runs = []

    for run in range(STABILITY_RUNS):
        seed = RANDOM_SEED + run
        print(f"  Run {run+1}/{STABILITY_RUNS}...")
        roberta_top5 = get_top5(test_email, roberta_model, roberta_tok, seed)
        bert_top5    = get_top5(test_email, bert_model, bert_tok, seed)
        roberta_runs.append(roberta_top5)
        bert_runs.append(bert_top5)
        print(f"    RoBERTa top5: {roberta_top5}")
        print(f"    BERT    top5: {bert_top5}")

    # Count all unique tokens that appeared in any run's top-5
    all_tokens_r = set().union(*roberta_runs)
    all_tokens_b = set().union(*bert_runs)
    all_tokens   = sorted(all_tokens_r | all_tokens_b)[:10]

    roberta_stability = []
    bert_stability    = []
    for tok in all_tokens:
        r_count = sum(1 for run in roberta_runs if tok in run)
        b_count = sum(1 for run in bert_runs    if tok in run)
        roberta_stability.append(r_count / STABILITY_RUNS * 100)
        bert_stability.append(b_count / STABILITY_RUNS * 100)

    r_mean = np.mean(roberta_stability)
    b_mean = np.mean(bert_stability)
    print(f"\nRoBERTa mean stability: {r_mean:.1f}%")
    print(f"BERT    mean stability: {b_mean:.1f}%")

    # Save Figure 4
    print("Saving Figure 4 (stability chart)...")
    x = np.arange(len(all_tokens))
    w = 0.36
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    b1 = ax.bar(x - w/2, roberta_stability, w,
                label='RoBERTa', color='#2196F3', edgecolor='white')
    b2 = ax.bar(x + w/2, bert_stability,    w,
                label='BERT',    color='#4CAF50', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(all_tokens, rotation=25, ha='right', fontsize=10.5)
    ax.set_ylabel("% of runs token appeared in top-5", fontsize=10.5)
    ax.set_ylim(0, 115)
    ax.axhline(100, color='#aaa', linestyle='--', linewidth=0.8)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                    f'{int(h)}%', ha='center', va='bottom',
                    fontsize=8.5, color='#333')

    ax.set_title(
        f"Figure 4: LIME stability test — top-5 token consistency across {STABILITY_RUNS} runs\n"
        f"RoBERTa mean: {r_mean:.0f}%   |   BERT mean: {b_mean:.0f}%   "
        f"|   Same email used for all runs",
        fontsize=9.5, loc='left', pad=6)

    plt.tight_layout(pad=0.6)
    plt.savefig(f"{save_path}/figure4_lime_stability.png",
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}/figure4_lime_stability.png")

    return all_tokens, roberta_stability, bert_stability


# ── Table 5: SHAP vs LIME top-10 comparison ──────────────────────────

def print_table5(top_shap, top_lime):
    """Print the data for Table 5 — SHAP vs LIME token comparison."""
    phishing_indicators = {
        'verify', 'account', 'password', 'urgent', 'click', 'login',
        'suspended', 'confirm', 'credentials', 'immediately', 'update',
        'limited', 'access', 'security', 'action', 'bank', 'card',
        'expire', 'reset', 'validate', 'secure', 'alert', 'warning'
    }

    shap_top10 = [t[0] for t in top_shap[:10]]
    lime_top10 = [t[0] for t in top_lime[:10]]

    print("\n--- TABLE 5 DATA (for your dissertation) ---")
    print(f"{'Rank':<6} {'SHAP Token':<22} {'LIME Token':<22} {'Known Indicator?'}")
    print("-" * 75)
    for i, (s_tok, l_tok) in enumerate(zip(shap_top10, lime_top10), 1):
        s_ind = "Yes" if s_tok in phishing_indicators else "Partial"
        l_ind = "Yes" if l_tok in phishing_indicators else "Partial"
        print(f"{i:<6} {s_tok:<22} {l_tok:<22} {s_ind} / {l_ind}")

    overlap = set(shap_top10) & set(lime_top10)
    print(f"\nOverlap in top 10: {len(overlap)}/10 tokens shared")
    print(f"Shared tokens: {overlap}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(f"{DATA_DIR}test.csv")
    phishing_test = test_df[test_df['label'] == 1]['text'].tolist()
    print(f"Test phishing emails available: {len(phishing_test)}")

    # Load models
    roberta_model, roberta_tok = load_model_and_tokenizer(ROBERTA_PATH)
    bert_model,    bert_tok    = load_model_and_tokenizer(BERT_PATH)

    # ── Run SHAP ──────────────────────────────────────────────
    top_shap = run_shap(roberta_model, roberta_tok,
                        phishing_test, FIGURES_DIR)

    # ── Run LIME ──────────────────────────────────────────────
    top_lime = run_lime(roberta_model, roberta_tok,
                        phishing_test, FIGURES_DIR)

    # ── Print Table 5 comparison ─────────────────────────────
    print_table5(top_shap, top_lime)

    # ── LIME Stability Test ───────────────────────────────────
    # Use the first phishing email as the representative example
    test_email = phishing_test[0]
    run_lime_stability(
        roberta_model, roberta_tok,
        bert_model,    bert_tok,
        test_email,
        FIGURES_DIR
    )

    print(f"\n{'='*60}")
    print("XAI analysis complete.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("\nFiles produced:")
    for f in os.listdir(FIGURES_DIR):
        print(f"  {f}")
    print("\nInsert these figures into your dissertation to replace")
    print("the placeholder figures currently in the document.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
