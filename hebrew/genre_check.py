"""
How much of the poems' antiquity could be genre rather than date?

Residuals on the anchored corpus are structured by genre.  If poetry as such is
placed early, part of the antiquity assigned to the archaic poems is an artifact.
The corpus contains one securely dated poetic book, so no correction can be
estimated reliably -- but the robustness of the conclusion to applying one
anyway can be, and that is the useful quantity.
"""
import json
import numpy as np, pandas as pd

B = pd.read_csv("/home/claude/final_lobo_books.csv")
D = pd.read_csv("/home/claude/big_features_500.csv")
gen = D.groupby("unit").genre.first()
B["genre"] = B.book.map(gen)
G = (B.groupby("genre").resid.agg(n="size", mean="mean", min="min", max="max")
       .sort_values("mean"))
print("residual (true minus estimated) by genre; negative = placed too early\n")
print(G.to_string())
G.to_csv("/home/claude/genre_residuals.csv")

poetry = float(B.loc[B.genre == "poetry", "resid"].mean())
print(f"\npoetry residual: {poetry:+.0f} yr  (from {int((B.genre=='poetry').sum())} book)")

P = pd.read_csv("/home/claude/poem_predictions.csv").set_index("unit")
resid = (B.truth - B.pred).values
EX = 586
rows = []
for u, name in [("SongDeborah_poem", "Song of Deborah"),
                ("SongSea_poem", "Song of the Sea"),
                ("SongMoses_poem", "Song of Moses")]:
    raw = float(P.loc[u].pred)
    adj = raw + poetry                      # remove the estimated genre bias
    p_raw = float(np.mean((raw + resid) < EX))
    p_adj = float(np.mean((adj + resid) < EX))
    need = raw - EX                          # correction needed to cross the exile
    rows.append(dict(unit=name, raw=round(raw), adjusted=round(adj),
                     p_post_raw=round(p_raw, 2), p_post_adj=round(p_adj, 2),
                     margin_to_exile=round(need)))
    print(f"  {name:<18} {raw:4.0f} -> {adj:4.0f} BCE   "
          f"P(post-exilic) {p_raw:.2f} -> {p_adj:.2f}   "
          f"margin to 586 BCE: {need:+.0f} yr")

R = pd.DataFrame(rows)
R.to_csv("/home/claude/genre_adjusted_poems.csv", index=False)
sea = R[R.unit == "Song of the Sea"].iloc[0]
ratio = sea.margin_to_exile / abs(poetry)
print(f"\n  A genre correction would have to be {ratio:.1f} times the observed "
      f"poetry residual\n  to move the Song of the Sea across the exile.")
json.dump(dict(poetry_resid=poetry,
               narrative_resid=float(B.loc[B.genre == "narrative", "resid"].mean()),
               prophecy_resid=float(B.loc[B.genre == "prophecy", "resid"].mean()),
               n_prophecy=int((B.genre == "prophecy").sum()),
               sea_raw=int(sea.raw), sea_adj=int(sea.adjusted),
               sea_p_adj=float(sea.p_post_adj),
               deb_adj=int(R[R.unit == "Song of Deborah"].iloc[0].adjusted),
               deb_p_adj=float(R[R.unit == "Song of Deborah"].iloc[0].p_post_adj),
               sea_margin=int(sea.margin_to_exile), ratio=float(ratio)),
          open("/home/claude/genre_check.json", "w"), indent=2)
print("\nwrote genre_residuals.csv, genre_adjusted_poems.csv, genre_check.json")
