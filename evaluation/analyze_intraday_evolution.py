# analyze_intraday_evolution.py
"""
Analyse the evolution of the German continuous intraday market (2019-2025).

Computes yearly liquidity and volatility metrics directly from the
`transactions_intraday_de` table by issuing server-side aggregation queries,
so that no raw transaction data is transferred to the client.

Output:
  - CSV table with one row per year
  - LaTeX table (booktabs format) ready to be \\input into the thesis

Metrics per year:
  1. Total traded volume (TWh)
  2. Average daily traded volume (GWh)
  3. Number of trades (millions)
  4. Average trade size (MWh)
  5. Intraday price volatility: mean of per-delivery volume-weighted standard
     deviation of trade prices (EUR/MWh)
  6. Share of volume traded in the last 60 minutes before delivery (%)
  7. Share of volume in Quarter-Hour products (%)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import pandas as pd
import psycopg2
from loguru import logger
from dotenv import load_dotenv


# --- Database configuration -------------------------------------------------

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


PathLike = Union[str, Path]


# --- Configuration ---------------------------------------------------------

# Analysis window (inclusive start, exclusive end)
START_YEAR = 2019
END_YEAR = 2026  # exclusive => last full year analysed is 2025

# Filter: count each trade exactly once. The DB stores each transaction twice
# (once as BUY, once as SELL); using side = 'BUY' yields per-trade aggregates.
SIDE_FILTER = "BUY"

# Output paths
OUTPUT_DIR = Path("./outputs/intraday_evolution")
CSV_PATH = OUTPUT_DIR / "intraday_market_evolution.csv"
TEX_PATH = OUTPUT_DIR / "intraday_market_evolution.tex"


# --- SQL queries -----------------------------------------------------------

# Liquidity and structure metrics (one row per year)
SQL_LIQUIDITY = f"""
WITH base AS (
    SELECT
        EXTRACT(YEAR FROM deliverystart AT TIME ZONE 'Europe/Berlin')::int AS year,
        (deliverystart AT TIME ZONE 'Europe/Berlin')::date AS delivery_day,
        product,
        volume,
        trade_count,
        EXTRACT(EPOCH FROM (deliverystart - executiontime)) / 60.0
            AS minutes_to_delivery
    FROM transactions_intraday_de
    WHERE deliverystart >= '{START_YEAR}-01-01'
      AND deliverystart <  '{END_YEAR}-01-01'
      AND side = '{SIDE_FILTER}'
)
SELECT
    year,
    SUM(volume)                                              AS total_volume_mwh,
    SUM(trade_count)                                         AS total_trades,
    SUM(CASE WHEN minutes_to_delivery <= 60
             THEN volume ELSE 0 END)                         AS volume_last_60min,
    SUM(CASE WHEN product ILIKE '%%Quarter_Hour%%'
             THEN volume ELSE 0 END)                         AS volume_qh,
    COUNT(DISTINCT delivery_day)                             AS active_days
FROM base
GROUP BY year
ORDER BY year;
"""

# Volume-weighted price volatility per delivery period, then averaged per year.
#
# Logic:
#   - For each (year, deliverystart) compute the volume-weighted variance of
#     trade prices:  Var_w(p) = ( sum(v * p^2) / sum(v) ) - ( sum(v * p) / sum(v) )^2
#   - Std_w(p) = sqrt(Var_w(p))
#   - Mean of Std_w(p) per year (equal-weighted across delivery periods)
#
# To avoid degenerate single-trade buckets, only delivery periods with at least
# MIN_TRADES_PER_DELIVERY underlying aggregated rows are kept.
MIN_TRADES_PER_DELIVERY = 5

SQL_VOLATILITY = f"""
WITH base AS (
    SELECT
        EXTRACT(YEAR FROM deliverystart AT TIME ZONE 'Europe/Berlin')::int AS year,
        deliverystart,
        weighted_avg_price AS price,
        volume,
        trade_count
    FROM transactions_intraday_de
    WHERE deliverystart >= '{START_YEAR}-01-01'
      AND deliverystart <  '{END_YEAR}-01-01'
      AND side = '{SIDE_FILTER}'
      AND volume > 0
),
per_delivery AS (
    SELECT
        year,
        deliverystart,
        SUM(volume)                                  AS sum_v,
        SUM(volume * price)                          AS sum_vp,
        SUM(volume * price * price)                  AS sum_vp2,
        SUM(trade_count)                             AS n_trades
    FROM base
    GROUP BY year, deliverystart
),
per_delivery_std AS (
    SELECT
        year,
        deliverystart,
        CASE
            WHEN sum_v > 0
            THEN sqrt(
                GREATEST(
                    (sum_vp2 / sum_v) - POWER(sum_vp / sum_v, 2),
                    0
                )
            )
            ELSE NULL
        END AS vw_std
    FROM per_delivery
    WHERE n_trades >= {MIN_TRADES_PER_DELIVERY}
)
SELECT
    year,
    AVG(vw_std)         AS mean_vw_std,
    COUNT(*)            AS n_delivery_periods
FROM per_delivery_std
WHERE vw_std IS NOT NULL
GROUP BY year
ORDER BY year;
"""


# --- Helpers ---------------------------------------------------------------

def fetch_dataframe(query: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a DataFrame."""
    logger.info("Executing query against {}", POSTGRES_DB_HOST)
    with psycopg2.connect(CONNECTION) as conn:
        df = pd.read_sql_query(query, conn)
    logger.info(" -> {} rows returned", len(df))
    return df


def build_summary_table(
    df_liq: pd.DataFrame,
    df_vol: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine the liquidity and volatility aggregates into one summary DataFrame
    with the final reporting units.
    """
    df = df_liq.merge(df_vol, on="year", how="left")

    # Derived metrics
    df["total_volume_twh"] = df["total_volume_mwh"] / 1_000_000
    df["avg_daily_volume_gwh"] = (df["total_volume_mwh"] / df["active_days"]) / 1_000
    df["total_trades_millions"] = df["total_trades"] / 1_000_000
    df["avg_trade_size_mwh"] = df["total_volume_mwh"] / df["total_trades"]
    df["late_trading_share_pct"] = (
        df["volume_last_60min"] / df["total_volume_mwh"] * 100
    )
    df["qh_volume_share_pct"] = (
        df["volume_qh"] / df["total_volume_mwh"] * 100
    )

    # Keep reporting columns in a sensible order
    out = df[[
        "year",
        "total_volume_twh",
        "avg_daily_volume_gwh",
        "total_trades_millions",
        "avg_trade_size_mwh",
        "mean_vw_std",
        "late_trading_share_pct",
        "qh_volume_share_pct",
    ]].copy()

    out.columns = [
        "Year",
        "Total volume (TWh)",
        "Avg. daily volume (GWh)",
        "Trades (million)",
        "Avg. trade size (MWh)",
        "Intraday price vol. (EUR/MWh)",
        "Volume in last 60 min (%)",
        "Quarter-Hour share (%)",
    ]

    return out


def format_latex_table(df: pd.DataFrame) -> str:
    """
    Render the summary DataFrame as a booktabs-style LaTeX table.
    """
    # Per-column number formatting
    fmt = {
        "Year": "{:d}".format,
        "Total volume (TWh)": "{:.1f}".format,
        "Avg. daily volume (GWh)": "{:.1f}".format,
        "Trades (million)": "{:.1f}".format,
        "Avg. trade size (MWh)": "{:.3f}".format,
        "Intraday price vol. (EUR/MWh)": "{:.2f}".format,
        "Volume in last 60 min (%)": "{:.1f}".format,
        "Quarter-Hour share (%)": "{:.1f}".format,
    }

    df_fmt = df.copy()
    for col, f in fmt.items():
        df_fmt[col] = df_fmt[col].map(
            lambda x, f=f: "--" if pd.isna(x) else f(x)
        )

    # Build LaTeX manually for full control over column spec and headers
    n_cols = len(df_fmt.columns)
    col_spec = "l" + "r" * (n_cols - 1)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Evolution of the German continuous intraday electricity "
        r"market, 2019--2025. Metrics computed from EPEX SPOT transaction "
        r"data for all continuous intraday products (hourly and quarter-hourly)."
        r"}"
    )
    lines.append(r"\label{tab:intraday_market_evolution}")
    lines.append(r"\small")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(df_fmt.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df_fmt.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


# --- Main ------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_liq = fetch_dataframe(SQL_LIQUIDITY)
    df_vol = fetch_dataframe(SQL_VOLATILITY)

    summary = build_summary_table(df_liq, df_vol)

    # Save CSV (full precision for downstream use)
    summary.to_csv(CSV_PATH, index=False)
    logger.info("Saved CSV: {}", CSV_PATH)

    # Save LaTeX table
    tex_str = format_latex_table(summary)
    TEX_PATH.write_text(tex_str, encoding="utf-8")
    logger.info("Saved LaTeX table: {}", TEX_PATH)

    # Console preview
    print("\nIntraday market evolution summary:\n")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:.3f}".format,
    ):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()