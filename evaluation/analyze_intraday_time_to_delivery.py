# analyze_intraday_time_to_delivery.py
"""
Year × time-to-delivery distribution of traded volume for the German
continuous intraday Quarter-Hour market (2019–2025).

Aggregates server-side on PostgreSQL (`transactions_intraday_de`); only
aggregates are transferred. Outputs CSV, LaTeX table, and heatmap figures under
`./outputs/intraday_evolution/`.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns
from dotenv import load_dotenv
from loguru import logger
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


# --- Database configuration (same pattern as precompute_vwaps / evolution) ---

load_dotenv()

PASSWORD = os.getenv("SQL_PASSWORD")
password_for_url = f":{PASSWORD}" if PASSWORD else ""

THESIS_DB_NAME = os.getenv("POSTGRES_DB_NAME")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_DB_HOST = os.getenv("POSTGRES_DB_HOST")

if not all([THESIS_DB_NAME, POSTGRES_USER, POSTGRES_DB_HOST]):
    raise RuntimeError(
        "Missing required database environment variables "
        "(POSTGRES_DB_NAME, POSTGRES_USER, POSTGRES_DB_HOST)."
    )

CONNECTION = (
    f"postgres://{POSTGRES_USER}{password_for_url}@{POSTGRES_DB_HOST}/{THESIS_DB_NAME}"
)


# --- Analysis configuration -------------------------------------------------

START_YEAR = 2019
END_YEAR = 2026  # exclusive window end => full calendar years through 2025
SIDE_FILTER = "BUY"

# Quarter-hour products only (EPEX naming)
PRODUCT_FILTER_SQL = """
    (
        product ILIKE '%%Intraday_Quarter_Hour_Power%%'
        OR product ILIKE '%%XBID_Quarter_Hour_Power%%'
    )
"""

# SQL column aliases (longest horizon first; last bucket merges 0–5 and 5–15 min)
BUCKET_SQL_ALIASES: tuple[str, ...] = (
    "bucket_gt480",
    "bucket_240_480",
    "bucket_120_240",
    "bucket_60_120",
    "bucket_30_60",
    "bucket_15_30",
    "bucket_0_15",
)

# Human-readable column names for CSV / plots (left = early trading)
BUCKET_LABELS_DISPLAY: tuple[str, ...] = (
    ">480 min",
    "240–480",
    "120–240",
    "60–120",
    "30–60",
    "15–30",
    "0–15",
)

# LaTeX table headers (en-dash as `--`; first bucket in math mode)
BUCKET_LABELS_LATEX: tuple[str, ...] = (
    r"$>480$",
    r"240--480",
    r"120--240",
    r"60--120",
    r"30--60",
    r"15--30",
    r"0--15",
)

OUTPUT_DIR = Path("./outputs/intraday_evolution")
CSV_PATH = OUTPUT_DIR / "intraday_time_to_delivery.csv"
TEX_PATH = OUTPUT_DIR / "intraday_time_to_delivery.tex"
HEATMAP_PDF_PATH = OUTPUT_DIR / "intraday_time_to_delivery_heatmap.pdf"
HEATMAP_PNG_PATH = OUTPUT_DIR / "intraday_time_to_delivery_heatmap.png"

# Heatmap layout (diverging scale similar to RdBu_r: low = blue, high = red)
FIG_SIZE_INCHES = (10, 4.5)
FIG_DPI = 200
HEATMAP_CMAP = "RdBu_r"
AXIS_TICK_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 16

SQL_BUCKET_AGG = f"""
WITH base AS (
    SELECT
        EXTRACT(YEAR FROM deliverystart AT TIME ZONE 'Europe/Berlin')::int AS year,
        EXTRACT(EPOCH FROM (deliverystart - executiontime)) / 60.0 AS minutes_to_delivery,
        volume
    FROM transactions_intraday_de
    WHERE deliverystart >= '{START_YEAR}-01-01'
      AND deliverystart < '{END_YEAR}-01-01'
      AND side = '{SIDE_FILTER}'
      AND volume > 0
      AND executiontime IS NOT NULL
      AND deliverystart IS NOT NULL
      AND deliverystart >= executiontime
      AND {PRODUCT_FILTER_SQL.strip()}
)
SELECT
    year,
    SUM(CASE WHEN minutes_to_delivery > 480 THEN volume ELSE 0 END) AS bucket_gt480,
    SUM(
        CASE
            WHEN minutes_to_delivery > 240 AND minutes_to_delivery <= 480
            THEN volume ELSE 0 END
    ) AS bucket_240_480,
    SUM(
        CASE
            WHEN minutes_to_delivery > 120 AND minutes_to_delivery <= 240
            THEN volume ELSE 0 END
    ) AS bucket_120_240,
    SUM(
        CASE
            WHEN minutes_to_delivery > 60 AND minutes_to_delivery <= 120
            THEN volume ELSE 0 END
    ) AS bucket_60_120,
    SUM(
        CASE
            WHEN minutes_to_delivery > 30 AND minutes_to_delivery <= 60
            THEN volume ELSE 0 END
    ) AS bucket_30_60,
    SUM(
        CASE
            WHEN minutes_to_delivery > 15 AND minutes_to_delivery <= 30
            THEN volume ELSE 0 END
    ) AS bucket_15_30,
    SUM(
        CASE
            WHEN minutes_to_delivery >= 0 AND minutes_to_delivery <= 15
            THEN volume ELSE 0 END
    ) AS bucket_0_15,
    SUM(volume) AS total_volume
FROM base
GROUP BY year
ORDER BY year;
"""


def fetch_bucket_volumes() -> pd.DataFrame:
    """Run the aggregation query and return one row per year with bucket volumes."""
    logger.info("Executing time-to-delivery aggregation against {}", POSTGRES_DB_HOST)
    with psycopg2.connect(CONNECTION) as conn:
        df = pd.read_sql_query(SQL_BUCKET_AGG, conn)
    logger.info(" -> {} rows returned", len(df))
    df["year"] = df["year"].astype(int)
    return df


def volumes_to_percentage_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert absolute bucket volumes to percentage shares of yearly `total_volume`.

    Returns a DataFrame indexed by year with only the seven bucket columns (%).
    """
    total = df["total_volume"].replace(0, np.nan)
    out = df[list(BUCKET_SQL_ALIASES)].div(total, axis=0) * 100.0
    out = out.fillna(0.0)
    out.columns = list(BUCKET_LABELS_DISPLAY)
    out.index = df["year"].astype(int)
    out.index.name = "year"
    return out


def format_latex_table(df_pct: pd.DataFrame) -> str:
    """
    Render the percentage table as a booktabs-style LaTeX fragment for \\input.
    """
    n_bucket = len(BUCKET_LABELS_LATEX)
    col_spec = "l" + "r" * n_bucket

    cap = (
        r"Distribution of traded volume across time-to-delivery buckets "
        r"(in minutes before delivery start) for the German continuous intraday "
        r"Quarter-Hour market. Values in \% of yearly volume. Each trade is counted "
        r"on one market side only."
    )

    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{cap}}}")
    lines.append(r"\label{tab:intraday_time_to_delivery}")
    lines.append(r"\small")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = "Year & " + " & ".join(BUCKET_LABELS_LATEX) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for year, row in df_pct.iterrows():
        cells = [f"{year:d}"] + [f"{v:.1f}" for v in row.values]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def _annotation_text_color(rgba: tuple[float, ...]) -> str:
    """Pick black or white annotation text given an RGBA facecolor from the colormap."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 0.55 else "black"


def plot_time_to_delivery_heatmap(df_pct: pd.DataFrame, output_pdf: Path, output_png: Path) -> None:
    """
    Save a heatmap of percentage shares (years × buckets) as PDF and PNG.

    Years run from earliest at the top to latest at the bottom; buckets run
    from longest time before delivery (left) to closest (right).
    """
    # Ensure row order: 2019 at top → ascending year index in DataFrame
    data = df_pct.sort_index(ascending=True)
    arr = data.values.astype(float)
    nrow, ncol = arr.shape

    norm = mcolors.Normalize(vmin=float(np.nanmin(arr)), vmax=float(np.nanmax(arr)))
    cmap = plt.get_cmap(HEATMAP_CMAP)

    fig, ax = plt.subplots(figsize=FIG_SIZE_INCHES)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.12)
    sns.heatmap(
        data,
        ax=ax,
        cbar_ax=cax,
        cmap=HEATMAP_CMAP,
        norm=norm,
        cbar_kws={"label": "Share (%)"},
        linewidths=0.5,
        linecolor="white",
        annot=False,
    )
    ax.set_xlabel(
        "Time before delivery (minutes)", fontsize=AXIS_LABEL_FONTSIZE
    )
    ax.set_ylabel("Year", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    cax.tick_params(labelsize=AXIS_TICK_FONTSIZE)
    cax.yaxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)

    for i in range(nrow):
        for j in range(ncol):
            val = arr[i, j]
            rgba = cmap(norm(val))
            color = _annotation_text_color(rgba)
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight", dpi=FIG_DPI)
    fig.savefig(output_png, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)
    logger.info("Saved heatmap: {} and {}", output_pdf, output_png)


def _log_row_sum_check(df_pct: pd.DataFrame, tol: float = 1.0) -> None:
    """Log a warning if any year's bucket percentages deviate strongly from 100%."""
    sums = df_pct.sum(axis=1)
    bad = sums[(sums < 100.0 - tol) | (sums > 100.0 + tol)]
    if not bad.empty:
        logger.warning("Yearly percentage row sums outside [{}, {}]: {}", 100 - tol, 100 + tol, bad.to_dict())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = fetch_bucket_volumes()
    df_pct = volumes_to_percentage_shares(df_raw)

    _log_row_sum_check(df_pct)

    # CSV: one row per year, seven bucket columns in %
    df_pct.reset_index().to_csv(CSV_PATH, index=False)
    logger.info("Saved CSV: {}", CSV_PATH)

    TEX_PATH.write_text(format_latex_table(df_pct), encoding="utf-8")
    logger.info("Saved LaTeX table: {}", TEX_PATH)

    plot_time_to_delivery_heatmap(df_pct, HEATMAP_PDF_PATH, HEATMAP_PNG_PATH)

    print("\nTime-to-delivery volume shares (% of yearly volume):\n")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df_pct.to_string(float_format=lambda x: f"{x:.3f}"))

    sums = df_pct.sum(axis=1)
    print("\nRow sums (%):", sums.to_dict())


if __name__ == "__main__":
    main()
