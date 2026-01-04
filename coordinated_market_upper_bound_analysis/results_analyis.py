import pandas as pd
from pathlib import Path

# 1) Oracle Summaries laden
oracle_dir = Path("../results")  # anpassen
files = sorted(oracle_dir.glob("summary_*.csv"))

oracle = pd.concat(
    (pd.read_csv(f, parse_dates=["day"]) for f in files),
    ignore_index=True
)

# Vereinheitlichen: nur Datum
oracle["day"] = pd.to_datetime(oracle["day"], errors="coerce").dt.normalize()

# 2) RI_base aus combo_id==0 ziehen
ri_base = (
    oracle.loc[oracle["combo_id"] == 0, ["day", "profit"]]
    .rename(columns={"profit": "RI_base"})
    .drop_duplicates(subset=["day"])
)

# 3) Alle "synthetic" Runs (ohne combo 0)
syn = oracle.loc[oracle["combo_id"] != 0].copy()

# 4) Baseline an synthetic dranhängen
syn = syn.merge(ri_base, on="day", how="left")

# Prozentualer uplift je combo
syn["uplift_pct"] = (syn["profit"] - syn["RI_base"]) / syn["RI_base"] * 100

# 5) Tages-Übersicht bauen
summary = (
    syn.groupby("day")
    .apply(lambda g: pd.Series({
        "RI_base": g["RI_base"].iloc[0],
        "Oracle_max": g["profit"].max(),
        "Oracle_max_pct": (g["profit"].max() - g["RI_base"].iloc[0]) / g["RI_base"].iloc[0] * 100,
        "n_better_than_base": (g["profit"] > g["RI_base"]).sum(),
        "mean_uplift_pct_if_better": g.loc[g["profit"] > g["RI_base"], "uplift_pct"].mean(),
        "median_uplift_pct_if_better": g.loc[g["profit"] > g["RI_base"], "uplift_pct"].median(),
    }))
    .reset_index()
)

# Optional: NaNs (falls kein Run besser ist) auf 0 setzen
summary[["mean_uplift_pct_if_better", "median_uplift_pct_if_better"]] = (
    summary[["mean_uplift_pct_if_better", "median_uplift_pct_if_better"]].fillna(0.0)
)

# Speichern
out_path = oracle_dir / "oracle_overview_by_day.csv"
summary.to_csv(out_path, index=False)

print("Saved:", out_path)
summary.head()

