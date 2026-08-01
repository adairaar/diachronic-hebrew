"""
07_visualizations.py
====================
Generates the full visualization suite for the Greek diachronic analysis.
Mirrors the Hebrew pipeline's visual style.

Plots produced
--------------
  01_feature_trends.png        — Top 12 grammatical features vs. date (scatter + trend line)
  02_feature_trends_ngram.png  — Top 12 n-gram features vs. date
  03_pca_stylometric.png       — 4-panel PCA: PC1/PC2 by date and genre, PC1 vs date, scree
  04_feature_heatmap.png       — Z-scored feature heatmap (entries × robust features)
  05_dating_summary.png        — Dotplot: MAP dates + CI vs. scholarly consensus (training)
  06_holdout_validation.png    — Already produced by 06_holdout_validation.py; regenerated here
  07_corpus_timeline.png       — Timeline of corpus texts colored by genre

Usage
-----
    python 07_visualizations.py [--dpi 150]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    HAS_MPL = True
except ImportError:
    print("matplotlib not installed — run: pip install matplotlib --break-system-packages")
    sys.exit(1)

HERE       = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR   = os.path.join(HERE, "data", "features")
RESULTS    = os.path.join(HERE, "results")
PLOTS_DIR  = os.path.join(HERE, "results", "plots")
POST_DIR   = os.path.join(HERE, "results", "posteriors")
MANIFEST   = os.path.join(HERE, "corpus_manifest.json")

GENRE_COLORS = {
    "prose_history"    : "#1565C0",   # blue
    "prose_oratory"    : "#6A1B9A",   # purple
    "prose_philosophy" : "#2E7D32",   # green
    "prose_science"    : "#E65100",   # orange
    "prose_narrative"  : "#AD1457",   # pink
}

def load_corpus():
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)

def load_feature_matrix():
    path = os.path.join(FEAT_DIR, "feature_matrix.csv")
    return pd.read_csv(path, index_col="id") if os.path.exists(path) else None

def load_robust_names():
    path = os.path.join(RESULTS, "robust_feature_names.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)

def load_dating_results():
    path = os.path.join(RESULTS, "dating_results.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def load_feature_scan():
    path = os.path.join(RESULTS, "feature_scan_robust.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


# ---------------------------------------------------------------------------
# Plot 1/2: Feature trend scatter plots
# ---------------------------------------------------------------------------

def plot_feature_trends(df_train: pd.DataFrame, robust_names: list[str],
                        family: str, n_plots: int, out_path: str, dpi: int):
    """Scatter + trend line for top-N features of a given family."""
    if family == "gram":
        feats = [f for f in robust_names if not f.startswith(("c3_","c4_","bg_"))]
    else:
        prefix = f"{family}_"
        feats  = [f for f in robust_names if f.startswith(prefix)]

    feats = feats[:n_plots]
    if not feats:
        print(f"  No robust {family} features to plot.")
        return

    ncols = 3
    nrows = (len(feats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.flatten()

    dates = df_train["date_ce"].values.astype(float)
    genres = df_train["genre"].values if "genre" in df_train.columns else ["unknown"] * len(df_train)

    for ax, feat in zip(axes, feats):
        if feat not in df_train.columns:
            ax.set_visible(False)
            continue
        vals  = df_train[feat].values.astype(float)
        color = [GENRE_COLORS.get(g, "#607D8B") for g in genres]

        ax.scatter(dates, vals, c=color, s=40, alpha=0.8, edgecolors="white", linewidths=0.3)

        # OLS trend line
        mask = ~np.isnan(vals)
        if mask.sum() >= 3:
            from scipy.stats import linregress
            slope, intercept, r, p, _ = linregress(dates[mask], vals[mask])
            x_fit = np.linspace(dates.min(), dates.max(), 100)
            ax.plot(x_fit, intercept + slope * x_fit, color="#D32F2F", lw=1.5,
                    label=f"r={r:.2f}, p={p:.3f}")
            ax.legend(fontsize=7, loc="best")

        short = feat.replace("_rate","").replace("_frac","").replace("c3_","3g:").replace("c4_","4g:").replace("bg_","bg:")
        ax.set_title(short[:30], fontsize=9)
        ax.set_xlabel("Date (CE)", fontsize=8)
        ax.axvline(0, color="gray", lw=0.5, ls=":")  # BCE/CE boundary

    # Hide unused subplots
    for ax in axes[len(feats):]:
        ax.set_visible(False)

    # Genre legend
    handles = [mpatches.Patch(color=c, label=g.replace("prose_",""))
               for g, c in GENRE_COLORS.items()]
    fig.legend(handles=handles, loc="lower right", fontsize=8, ncol=3)

    title_map = {"gram": "Grammatical", "c3": "Character 3-gram",
                 "c4": "Character 4-gram", "bg": "Word Bigram"}
    fig.suptitle(f"Greek Corpus: {title_map.get(family, family)} Feature Trends vs. Date",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: PCA
# ---------------------------------------------------------------------------

def plot_pca(df_train: pd.DataFrame, robust_names: list[str], out_path: str, dpi: int):
    feat_cols = [f for f in robust_names if f in df_train.columns][:40]
    if len(feat_cols) < 3:
        print("  Insufficient features for PCA.")
        return

    X = df_train[feat_cols].values.astype(float)
    # Impute NaN
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    # Standardize
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    # PCA via eigendecomposition (no sklearn)
    cov = np.cov(X_std.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    explained    = eigenvalues / eigenvalues.sum()

    PC = X_std @ eigenvectors[:, :2]

    dates  = df_train["date_ce"].values.astype(float)
    genres = df_train["genre"].values if "genre" in df_train.columns else ["unknown"] * len(df_train)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: PC1 vs PC2 colored by date
    norm  = Normalize(vmin=dates.min(), vmax=dates.max())
    cmap  = cm.RdYlBu_r
    sc = axes[0, 0].scatter(PC[:, 0], PC[:, 1], c=dates, cmap=cmap, s=60, alpha=0.85)
    plt.colorbar(sc, ax=axes[0, 0], label="Date (CE)")
    axes[0, 0].set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    axes[0, 0].set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    axes[0, 0].set_title("PC1 vs PC2 (colored by date)")
    for i, row in df_train.reset_index().iterrows():
        axes[0, 0].annotate(row["id"].split("_")[0][:8], (PC[i, 0], PC[i, 1]),
                            fontsize=6, alpha=0.6)

    # Panel 2: PC1 vs PC2 colored by genre
    for genre, color in GENRE_COLORS.items():
        mask = genres == genre
        axes[0, 1].scatter(PC[mask, 0], PC[mask, 1], color=color,
                           label=genre.replace("prose_",""), s=60, alpha=0.85)
    axes[0, 1].set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    axes[0, 1].set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    axes[0, 1].set_title("PC1 vs PC2 (colored by genre)")
    axes[0, 1].legend(fontsize=8)

    # Panel 3: PC1 vs date
    from scipy.stats import spearmanr
    rho, p = spearmanr(dates, PC[:, 0])
    axes[1, 0].scatter(dates, PC[:, 0], c=[GENRE_COLORS.get(g, "#607D8B") for g in genres],
                       s=60, alpha=0.85)
    m = np.polyfit(dates, PC[:, 0], 1)
    x_fit = np.linspace(dates.min(), dates.max(), 100)
    axes[1, 0].plot(x_fit, np.polyval(m, x_fit), "r--", lw=1.5)
    axes[1, 0].axvline(0, color="gray", lw=0.5, ls=":")
    axes[1, 0].set_xlabel("Date (CE)")
    axes[1, 0].set_ylabel(f"PC1 ({explained[0]*100:.1f}%)")
    axes[1, 0].set_title(f"PC1 vs Date  (Spearman ρ={rho:.3f}, p={p:.4f})")

    # Panel 4: Scree plot
    n_scree = min(14, len(explained))
    axes[1, 1].bar(range(1, n_scree + 1),
                   explained[:n_scree] * 100, color="#42A5F5", edgecolor="white")
    axes[1, 1].set_xlabel("Principal Component")
    axes[1, 1].set_ylabel("Variance Explained (%)")
    axes[1, 1].set_title("Scree Plot")

    fig.suptitle("Greek Corpus: PCA Stylometric Analysis", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Feature heatmap
# ---------------------------------------------------------------------------

def plot_feature_heatmap(df_train: pd.DataFrame, robust_names: list[str],
                         out_path: str, dpi: int):
    feat_cols = [f for f in robust_names if f in df_train.columns][:30]
    if not feat_cols:
        return

    df_sorted = df_train.sort_values("date_ce")
    X = df_sorted[feat_cols].values.astype(float)

    # Z-score per feature
    means = np.nanmean(X, axis=0)
    stds  = np.nanstd(X, axis=0) + 1e-10
    Z = (X - means) / stds
    Z = np.clip(Z, -3, 3)  # cap outliers

    fig, ax = plt.subplots(figsize=(14, max(6, len(df_sorted) * 0.35)))
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    plt.colorbar(im, ax=ax, label="Z-score", fraction=0.02)

    short_feats = [f.replace("_rate","").replace("_frac","")
                    .replace("c3_","3g:")[:20] for f in feat_cols]
    ax.set_xticks(range(len(feat_cols)))
    ax.set_xticklabels(short_feats, rotation=45, ha="right", fontsize=7)

    labels = df_sorted.reset_index()["id"].apply(lambda x: x.split("_")[0][:12]).tolist()
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(labels, fontsize=7)

    ax.set_title("Feature Z-score Heatmap (rows sorted by date, earliest top)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Plot 5: Dating summary dotplot
# ---------------------------------------------------------------------------

def plot_dating_summary(df_results: pd.DataFrame, out_path: str, dpi: int):
    if df_results.empty:
        print("  No dating results found.")
        return

    df_plot = df_results[~df_results["holdout"]].copy()
    df_plot = df_plot.sort_values("scholarly_date_ce")

    fig, ax = plt.subplots(figsize=(10, max(5, len(df_plot) * 0.38)))
    y_pos = np.arange(len(df_plot))

    for i, (_, row) in enumerate(df_plot.iterrows()):
        # 95% CI bar
        ax.barh(y_pos[i], row["ci95_hi"] - row["ci95_lo"],
                left=row["ci95_lo"], height=0.25, color="#90CAF9", alpha=0.6)
        # 68% CI bar
        ax.barh(y_pos[i], row["ci68_hi"] - row["ci68_lo"],
                left=row["ci68_lo"], height=0.25, color="#1565C0", alpha=0.8)
        # MAP point
        ax.scatter(row["map_date_ce"], y_pos[i], color="#D32F2F", s=50, zorder=5)
        # Scholarly date
        ax.scatter(row["scholarly_date_ce"], y_pos[i], color="black",
                   s=40, marker="D", zorder=5, alpha=0.7)

    # Axis and labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{r['author'][:22]}" for _, r in df_plot.iterrows()],
        fontsize=8
    )
    ax.axvline(0, color="gray", lw=0.8, ls="--", label="BCE/CE boundary")
    ax.set_xlabel("Date (CE; negative = BCE)")
    ax.set_title("Greek Corpus Dating: MAP + Credible Intervals vs. Scholarly Dates", fontsize=11)

    # Legend
    legend_elements = [
        mpatches.Patch(color="#90CAF9", label="95% CI"),
        mpatches.Patch(color="#1565C0", label="68% CI"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#D32F2F",
                   markersize=8, label="MAP date"),
        plt.Line2D([0],[0], marker="D", color="w", markerfacecolor="black",
                   markersize=7, label="Scholarly date"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Plot 7: Corpus timeline
# ---------------------------------------------------------------------------

def plot_corpus_timeline(corpus: list, out_path: str, dpi: int):
    fig, ax = plt.subplots(figsize=(14, 6))

    training = [c for c in corpus if not c["holdout"]]
    holdouts  = [c for c in corpus if c["holdout"]]
    all_texts = sorted(training + holdouts, key=lambda x: x["date_ce"])

    for i, c in enumerate(all_texts):
        color = GENRE_COLORS.get(c["genre"], "#607D8B")
        style = "solid" if not c["holdout"] else "dashed"
        lw    = 1.5 if not c["holdout"] else 2.5

        # Uncertainty bar
        ax.plot([c["date_ce"] - c["date_sigma"], c["date_ce"] + c["date_sigma"]],
                [i, i], color=color, lw=lw, ls=style, alpha=0.8)
        ax.scatter(c["date_ce"], i, color=color, s=50,
                   marker="o" if not c["holdout"] else "s", zorder=4)
        ax.text(c["date_ce"] + c["date_sigma"] + 5, i,
                f"{c['author'][:20]}", va="center", fontsize=7, color=color)

    ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("Date (CE; negative = BCE)", fontsize=10)
    ax.set_yticks([])
    ax.set_title("Greek Prose Corpus: Chronological Overview\n"
                 "(solid = training, dashed squares = holdout; bar = ±1σ date uncertainty)",
                 fontsize=11)

    handles = [mpatches.Patch(color=c, label=g.replace("prose_",""))
               for g, c in GENRE_COLORS.items()]
    handles += [plt.Line2D([0],[0], ls="dashed", color="gray", label="Holdout text")]
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Greek corpus visualizations.")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    corpus       = load_corpus()
    df_all       = load_feature_matrix()
    robust_names = load_robust_names()
    df_results   = load_dating_results()

    if df_all is None:
        print("feature_matrix.csv not found — run 03_feature_extraction.py first.")
        sys.exit(1)

    holdout_ids  = {c["id"] for c in corpus if c["holdout"]}
    df_train     = df_all[~df_all.index.isin(holdout_ids)].copy()

    print("Generating visualizations …\n")

    # 1. Grammatical feature trends
    print("1. Grammatical feature trends …")
    plot_feature_trends(df_train, robust_names, "gram", 12,
                        os.path.join(PLOTS_DIR, "01_feature_trends.png"), args.dpi)

    # 2. N-gram feature trends (char 3-grams)
    print("2. Character 3-gram feature trends …")
    plot_feature_trends(df_train, robust_names, "c3", 12,
                        os.path.join(PLOTS_DIR, "02_feature_trends_ngram.png"), args.dpi)

    # 3. PCA
    print("3. PCA stylometric analysis …")
    plot_pca(df_train, robust_names,
             os.path.join(PLOTS_DIR, "03_pca_stylometric.png"), args.dpi)

    # 4. Heatmap
    print("4. Feature heatmap …")
    plot_feature_heatmap(df_train, robust_names,
                         os.path.join(PLOTS_DIR, "04_feature_heatmap.png"), args.dpi)

    # 5. Dating summary
    print("5. Dating summary dotplot …")
    plot_dating_summary(df_results,
                        os.path.join(PLOTS_DIR, "05_dating_summary.png"), args.dpi)

    # 6. Holdout validation (regenerate from posteriors)
    print("6. Holdout validation plot …")
    # Delegate to 06_holdout_validation.py logic via import
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "holdout_val",
            os.path.join(HERE, "06_holdout_validation.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
    except Exception as e:
        print(f"  Could not regenerate holdout plot: {e}")

    # 7. Corpus timeline
    print("7. Corpus timeline …")
    plot_corpus_timeline(corpus,
                         os.path.join(PLOTS_DIR, "07_corpus_timeline.png"), args.dpi)

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
