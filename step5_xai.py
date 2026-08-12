def run_shap(model, tokenizer, phishing_emails, save_path):
    """
    Run SHAP Partition Explainer on a sample of phishing emails.
    Saves a token importance plot for the first email as Figure 2.
    Returns the global token ranking.

    FIX (Aug 2026): the original version estimated each token's box width
    with `len(token) * 0.011` instead of measuring the actual rendered text,
    which caused overlapping/garbled boxes for some tokens (e.g. "complete",
    "tax", "information" overlapping in the output). This version measures
    real text extents via the figure's renderer, so boxes never overlap
    regardless of font/token length.
    """
    print(f"\nRunning SHAP on {len(phishing_emails)} phishing emails...")

    def predict_fn(texts):
        return predict_proba(texts, model, tokenizer)

    masker = shap.maskers.Text(tokenizer=r"\W+")
    explainer = shap.Explainer(predict_fn, masker, output_names=["Legit", "Phishing"])
    shap_values = explainer(phishing_emails[:N_SHAP_SAMPLES], fixed_context=1)

    print("Saving Figure 2 (SHAP token map)...")

    FIG_WIDTH_IN = 16
    ROW_HEIGHT = 0.30
    FONT_SIZE = 11
    TOKEN_PAD_X = 0.006   # horizontal padding inside each box, in axes-fraction units
    TOKEN_GAP = 0.006     # gap between boxes
    LEFT_MARGIN = 0.01
    RIGHT_MARGIN = 0.98

    vals = shap_values[0, :, 1].values
    tokens = shap_values[0, :, 1].data
    max_abs_val = max(abs(v) for v in vals) + 1e-8

    # First pass: figure out how many rows we'll need so we can size the
    # figure correctly (avoids the vertical squashing that also happened
    # in the original version when many tokens wrapped).
    fig_probe, ax_probe = plt.subplots(figsize=(FIG_WIDTH_IN, 3))
    ax_probe.set_xlim(0, 1)
    ax_probe.set_ylim(0, 1)
    ax_probe.axis('off')
    fig_probe.canvas.draw()
    renderer = fig_probe.canvas.get_renderer()
    inv = ax_probe.transData.inverted()

    def measured_width(tok_str):
        t = ax_probe.text(0, 0, tok_str, fontsize=FONT_SIZE, fontfamily='monospace')
        bbox = t.get_window_extent(renderer=renderer)
        (x0, _), (x1, _) = inv.transform((bbox.x0, bbox.y0)), inv.transform((bbox.x1, bbox.y1))
        t.remove()
        return x1 - x0

    rows_needed = 1
    x = LEFT_MARGIN
    widths = []
    for tok in tokens:
        tok_str = str(tok).strip()
        if not tok_str:
            widths.append(0)
            continue
        w = measured_width(tok_str) + 2 * TOKEN_PAD_X
        widths.append(w)
        if x + w > RIGHT_MARGIN:
            rows_needed += 1
            x = LEFT_MARGIN
        x += w + TOKEN_GAP
    plt.close(fig_probe)

    fig_height = max(2.2, rows_needed * ROW_HEIGHT + 1.0)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('#FAFAFA')

    x, y = LEFT_MARGIN, 1.0 - ROW_HEIGHT * 0.7
    for tok, val, w in zip(tokens, vals, widths):
        tok_str = str(tok).strip()
        if not tok_str:
            continue

        intensity = min(abs(float(val)) / max_abs_val, 1.0)
        if float(val) > 0.01:
            color = (1.0, max(0.15, 1.0 - intensity * 0.85),
                     max(0.15, 1.0 - intensity * 0.85))
        else:
            color = (0.95, 0.95, 0.95)

        if x + w > RIGHT_MARGIN:
            x = LEFT_MARGIN
            y -= ROW_HEIGHT

        rect = plt.Rectangle((x, y - 0.11), w, 0.20, color=color, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y - 0.01, tok_str,
                ha='center', va='center',
                fontsize=FONT_SIZE, fontfamily='monospace',
                color='#1a0000' if float(val) > 0.1 else '#333333',
                zorder=3)
        x += w + TOKEN_GAP

    ax.text(LEFT_MARGIN, max(0.02, y - ROW_HEIGHT - 0.05),
            "Red = pushes towards PHISHING    Grey = near-neutral",
            fontsize=9, color='#555555')
    ax.set_title(
        "Figure 2: SHAP token-level importance — representative phishing email (RoBERTa)\n"
        "Darker red = stronger contribution to phishing prediction",
        fontsize=11, loc='left', pad=8)

    plt.tight_layout(pad=0.4)
    plt.savefig(f"{save_path}/figure2_shap_tokens.png",
                dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"  Saved: {save_path}/figure2_shap_tokens.png")

    # Aggregate top tokens across all SHAP explanations
    token_scores = {}
