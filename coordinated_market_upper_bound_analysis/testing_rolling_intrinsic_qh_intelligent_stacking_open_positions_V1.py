import os
import warnings
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import gurobipy as gp
from loguru import logger

# Suppress noisy warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) 

# Default path for precomputed VWAP matrices
from src.shared.config import PRECOMPUTED_VWAP_PATH



def adjust_prices_block(
    prices_qh: pd.DataFrame,
    execution_time: pd.Timestamp,
    discount_rate: float,
) -> pd.DataFrame:
    """
    Create two vectorized, discounted price columns based on time-to-delivery
    and a continuous discount rate:

    - price_sell_adj: effective price for selling
    - price_buy_adj:  effective price for buying (inverse discounting)

    Rules:
    - Base price column: 'price' (rounded to 2 decimals).
    - Discount factor depends on hours between execution_time and delivery.
    - If absolute time to delivery <= 1 hour: use original price.
    - Negative prices are discounted with inverse sign in exponent
      (to preserve economics of negative pricing).
    """
    out = prices_qh.copy()

    if "price" not in out.columns:
        raise ValueError("prices_qh must contain a 'price' column.")
    out["price"] = out["price"].round(2)

    # Compute time difference (in hours) between execution_time and each product
    idx = out.index.to_numpy(dtype="datetime64[ns]")
    exec_ts = np.datetime64(execution_time.tz_convert(None), "ns")
    hours = (idx - exec_ts) / np.timedelta64(1, "h")  # ndarray[float]

    price = out["price"].to_numpy(dtype=float)
    is_nan = np.isnan(price)

    # Sign rule so negative prices are handled correctly:
    # price < 0 => +1, else -1 in the exponent
    sign = np.where(price < 0, +1.0, -1.0)

    # Continuous discount factor per row
    factor = np.exp((discount_rate / 100.0) * sign * hours)

    # For deliveries within 1 hour: keep original price
    use_orig = hours <= 1.0

    # Sell side: multiply by factor (except within 1 hour)
    price_sell_adj = np.where(use_orig, price, price * factor)

    # Buy side: divide by factor (except within 1 hour)
    price_buy_adj = np.where(use_orig, price, price / factor)

    # Preserve NaNs
    price_sell_adj = np.where(is_nan, np.nan, price_sell_adj)
    price_buy_adj = np.where(is_nan, np.nan, price_buy_adj)

    out["price_sell_adj"] = np.round(price_sell_adj, 2)
    out["price_buy_adj"] = np.round(price_buy_adj, 2)
    return out


def get_net_trades(trades: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Aggregate trades to a full delivery day in 15-minute resolution.

    Returns a DataFrame (index: all quarter-hours of the delivery day) with:
    - sum_buy
    - sum_sell
    - net_buy  (max(sum_buy - sum_sell, 0))
    - net_sell (max(sum_sell - sum_buy, 0))
    """
    start = end_date - pd.Timedelta(hours=2)
    start = start.replace(hour=0, minute=0)
    end = start.replace(hour=23, minute=45)
    idx = pd.date_range(start, end, freq="15min")

    # Case 1: no trades -> all zeros
    if trades.empty:
        return pd.DataFrame(
            0.0,
            index=idx,
            columns=["sum_buy", "sum_sell", "net_buy", "net_sell"],
        )

    # Case 2: aggregate trades by product and side
    grouped = trades.groupby(["product", "side"])["quantity"].sum().unstack(
        fill_value=0
    )

    # Safe access for buy/sell columns
    grouped["sum_buy"] = grouped.get("buy", 0.0)
    grouped["sum_sell"] = grouped.get("sell", 0.0)

    grouped["net_buy"] = grouped["sum_buy"] - grouped["sum_sell"]
    grouped["net_sell"] = grouped["sum_sell"] - grouped["sum_buy"]

    # No negative net volumes
    grouped["net_buy"] = grouped["net_buy"].clip(lower=0.0)
    grouped["net_sell"] = grouped["net_sell"].clip(lower=0.0)

    grouped = grouped[["sum_buy", "sum_sell", "net_buy", "net_sell"]]

    # Reindex to full day, fill missing with 0
    return grouped.reindex(idx, fill_value=0.0)


def load_vwaps_for_day(
    current_day: pd.Timestamp,
    vwaps_base_path: str = PRECOMPUTED_VWAP_PATH,
) -> pd.DataFrame:
    """
    Load precomputed VWAP matrix for a given delivery day.

    - Index: execution_time_end (bucket end times)
    - Columns: delivery start times (product quarter-hours)
    Both in Europe/Berlin timezone.
    """
    fname = os.path.join(vwaps_base_path, f"vwaps_{current_day:%Y-%m-%d}.parquet")

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"VWAP parquet file for {current_day:%Y-%m-%d} not found: {fname}"
        )

    matrix = pd.read_parquet(fname)

    # Index: execution_time_end (bucket end)
    matrix.index = pd.to_datetime(matrix.index, utc=True).tz_convert("Europe/Berlin")

    # Columns: delivery start times
    matrix.columns = pd.to_datetime(matrix.columns, utc=True).tz_convert(
        "Europe/Berlin"
    )

    return matrix

def infer_bucket_size_minutes(vwaps_day: pd.DataFrame) -> int:
    """
    Infer bucket size (in minutes) from the VWAP matrix index spacing.
    """
    idx = vwaps_day.index.sort_values()
    if len(idx) < 2:
        raise ValueError("VWAP matrix has <2 rows; cannot infer bucket size.")

    deltas = idx.to_series().diff().dropna()
    # take the most frequent delta to be robust against missing buckets
    most_common = deltas.value_counts().idxmax()
    minutes = int(most_common.total_seconds() / 60)

    if minutes <= 0:
        raise ValueError(f"Inferred invalid bucket size: {minutes} minutes")

    return minutes


def get_vwap_from_precomputed(
    vwaps_day: pd.DataFrame,
    execution_time_end: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Extract one VWAP row from the daily VWAP matrix and convert it into a
    quarter-hourly price curve for the full delivery day.

    Returns a DataFrame with:
    - index: all quarter-hours of the delivery day
    - column: 'price'
    """
    end_date = end_date.tz_convert("Europe/Berlin")

    start_of_day = end_date - pd.Timedelta(hours=2)
    start_of_day = start_of_day.replace(hour=0, minute=0)
    end_of_day = start_of_day.replace(hour=23, minute=45)

    product_index = pd.date_range(
        start_of_day, end_of_day, freq="15min", tz="Europe/Berlin"
    )

    if execution_time_end not in vwaps_day.index:
        return pd.DataFrame(index=product_index, columns=["price"], dtype=float)

    row = vwaps_day.loc[execution_time_end]  # Series

    vwap = row.to_frame(name="price")
    vwap = vwap.reindex(product_index)

    return vwap


def build_battery_model(
    T: List[pd.Timestamp],
    cap: float,
    c_rate: float,
    roundtrip_eff: float,
) -> Tuple[gp.Model, Dict[str, Any], Dict[Any, gp.Constr], gp.Constr, gp.Constr, gp.Constr]:

    """
    Build a persistent Gurobi model for a single battery over a full delivery day.

    Variables (indexed by quarter-hour in T):
    - current_buy_qh, current_sell_qh: buy/sell volumes in each QH
    - battery_soc: state of charge
    - net_buy, net_sell: true charging/discharging flows
    - charge_sign: binary flag to prevent simultaneous charge+discharge
    - z, w: auxiliary variables for piecewise linear / big-M logic

    The model is built once and reused for each execution bucket by
    adjusting RHS and the objective.
    """
    efficiency = roundtrip_eff ** 0.5
    M = cap * c_rate

    m = gp.Model("battery_persistent")
    m.Params.OutputFlag = 0

    # Trade decision vars
    current_buy_qh = m.addVars(T, lb=0.0, name="current_buy_qh")
    current_sell_qh = m.addVars(T, lb=0.0, name="current_sell_qh")

    # NEW
    # Split total sell into two "channels":
    # - free sells: normal intraday trading volume
    # - constrained sells: volume that is counted as DA-long unwind and therefore subject to a spread gate + budget
    current_sell_free_qh = m.addVars(T, lb=0.0, name="current_sell_free_qh")
    current_sell_constr_qh = m.addVars(T, lb=0.0, name="current_sell_constr_qh")

    # NEW
    # Enforce sell split identity, keep orginal problem structure intact
    for i in T:
        m.addConstr(
            current_sell_qh[i]
            == current_sell_free_qh[i] + current_sell_constr_qh[i],
            name=f"SellSplit_{i}",
        )

    # Battery vars
    battery_soc = m.addVars(T, lb=0.0, name="battery_soc")
    net_buy = m.addVars(T, lb=0.0, name="net_buy")
    net_sell = m.addVars(T, lb=0.0, name="net_sell")
    charge_sign = m.addVars(T, vtype=gp.GRB.BINARY, name="charge_sign")

    # Aux vars (existing)
    z = m.addVars(T, lb=0.0, name="z")
    w = m.addVars(T, lb=0.0, name="w")

    # SOC dynamics over time
    prev = T[0]
    for i in T[1:]:
        m.addConstr(
            battery_soc[i]
            == battery_soc[prev]
            + net_buy[prev] * efficiency / 4.0
            - net_sell[prev] / 4.0 / efficiency,
            name=f"BatteryBalance_{i}",
        )
        prev = i

    # Start with empty battery
    m.addConstr(battery_soc[T[0]] == 0.0, name="InitialBatterySOC")

    # Time-independent constraints
    for i in T:
        # Capacity and power limits
        m.addConstr(battery_soc[i] <= cap, name=f"Cap_{i}")
        m.addConstr(net_buy[i] <= cap * c_rate, name=f"BuyRate_{i}")
        m.addConstr(net_sell[i] <= cap * c_rate, name=f"SellRate_{i}")

        # Cannot discharge more than current SOC
        m.addConstr(net_sell[i] / efficiency / 4.0 <= battery_soc[i],name=f"SellVsSOC_{i}",)

        # Big-M constraints to prevent simultaneous buy/sell
        m.addConstr(net_buy[i] <= M * charge_sign[i], name=f"NetBuyBigM_{i}")
        m.addConstr(net_sell[i] <= M * (1 - charge_sign[i]), name=f"NetSellBigM_{i}")

        # Auxiliary variable z for charging
        m.addConstr(z[i] <= charge_sign[i] * M, name=f"ZUpper_{i}")
        m.addConstr(z[i] <= net_buy[i], name=f"ZNetBuy_{i}")
        m.addConstr(z[i] >= net_buy[i] - (1 - charge_sign[i]) * M, name=f"ZLower_{i}")
        m.addConstr(z[i] >= 0.0, name=f"ZNonNeg_{i}")

        # Auxiliary variable w for discharging
        m.addConstr(w[i] <= (1 - charge_sign[i]) * M, name=f"WUpper_{i}")
        m.addConstr(w[i] <= net_sell[i], name=f"WNetSell_{i}")
        m.addConstr(w[i] >= net_sell[i] - charge_sign[i] * M, name=f"WLower_{i}")
        m.addConstr(w[i] >= 0.0, name=f"WNonNeg_{i}")

    # Netting constraints (RHS updated per bucket)
    netting_constr: Dict[Any, gp.Constr] = {}
    for i in T:
        c = m.addConstr(
            z[i] - w[i] - current_buy_qh[i] + current_sell_qh[i] == 0.0,
            name=f"Netting_{i}",
        )
        netting_constr[i] = c

    # Cycle constraint (RHS updated per bucket)
    max_cycles_constr = m.addConstr(
        gp.quicksum(net_buy[i] * efficiency / 4.0 for i in T) <= 0.0,
        name="MaxCycles",
    )

    # NEW
    # Total constrained sell energy in this bucket is limited by remaining DA-long inventory r_long (MWh)
    # RHS is updated in each rolling bucket solve
    da_long_budget_constr = m.addConstr(
        gp.quicksum(current_sell_constr_qh[i] / 4.0 for i in T) <= 0.0,
        name="DALongBudget",
    )
    # NEW
    # Free sells are limited by "free_mwh" = cap - r_long (MWh) to avoid selling more than the non-DA-reserved portion
    # RHS is updated per bucket; this enables gradual activation of free sells as r_long decreases
    free_sell_budget_constr = m.addConstr(
        gp.quicksum(current_sell_free_qh[i] / 4.0 for i in T) <= 0.0,
        name="FreeSellBudget",
    )


    m.setObjective(0.0, gp.GRB.MAXIMIZE)
    m.update()

    vars_dict = {
        "current_buy_qh": current_buy_qh,
        "current_sell_qh": current_sell_qh,
        "current_sell_free_qh": current_sell_free_qh,
        "current_sell_constr_qh": current_sell_constr_qh,
        "battery_soc": battery_soc,
        "net_buy": net_buy,
        "net_sell": net_sell,
        "charge_sign": charge_sign,
        "z": z,
        "w": w,
        "efficiency": efficiency,
    }

    return m, vars_dict, netting_constr, max_cycles_constr, da_long_budget_constr, free_sell_budget_constr

def solve_bucket_with_persistent_model(
    m: gp.Model,
    vars: Dict[str, Any],
    netting_constr: Dict[Any, gp.Constr],
    max_cycles_constr: gp.Constr,
    da_long_budget_constr: gp.Constr,
    free_sell_budget_constr: gp.Constr,
    prices_qh: pd.DataFrame,
    execution_time: pd.Timestamp,
    discount_rate: float,
    prev_net_trades: pd.DataFrame,
    da_net_trades: pd.DataFrame,
    allowed_cycles: float,
    p_da_buy: float | None,
    tau: float,
    r_long_remaining_mwh: float,
) -> Tuple[pd.DataFrame | None, pd.DataFrame, float, float]:
    """
    Solve one intraday bucket using a persistent battery model.

    - Updates RHS of netting and cycle constraints based on previous trades.
    - Rebuilds the objective from discounted prices.
    - Returns:
        results: quarter-hourly decision variables
        trades:  realized trades in this bucket
        objval:  objective value (profit)
      If no optimal solution is found, returns (None, empty_trades, 0.0).

    NEW:
    - da_long_budget_constr: energy budget constraint for constrained sells (RHS = r_long_remaining_mwh)
    - free_sell_budget_constr: energy budget constraint for free sells (RHS = intraday-only net-long, MWh)
    - p_da_buy, tau: define the spread gate threshold for constrained sells
    - r_long_remaining_mwh: rolling DA-long inventory state carried across buckets (MWh, effective after losses)
    
    Returns:
      results, trades, objval, r_long_remaining_mwh_new
    """
    T = list(prices_qh.index)

    current_buy_qh = vars["current_buy_qh"]
    current_sell_qh = vars["current_sell_qh"]
    current_sell_free_qh = vars["current_sell_free_qh"]
    current_sell_constr_qh = vars["current_sell_constr_qh"]

    net_buy = vars["net_buy"]
    net_sell = vars["net_sell"]
    charge_sign = vars["charge_sign"]

    # 1) Prepare discounted prices
    prices_qh_adj_all = adjust_prices_block(prices_qh, execution_time, discount_rate)
    prev_net_trades = prev_net_trades.reindex(prices_qh.index).fillna(0.0)
    da_net_trades = da_net_trades.reindex(prices_qh.index).fillna(0.0)

    eps = 0.01

    # 2) Set RHS of cycle constraint
    max_cycles_constr.RHS = allowed_cycles * 1.0  # *cap (if cap != 1)

    # NEW
    # 2b) Set RHS of DA-long budget constraint (energy, MWh)
    da_long_budget_constr.RHS = max(0.0, float(r_long_remaining_mwh))

    # NEW
    # Free-sell budget:
    # - If there are no DA trades and no DA-long inventory, fall back to the
    #   previous, effectively non-binding behaviour: cap - r_long (cap=1.0).
    #   This preserves the original pure-intraday RI baseline.
    # - Otherwise (DA present), use the tight intraday-only net-long budget so that
    #   free sells cannot silently unwind DA-long.
    if (
        float(r_long_remaining_mwh) <= 1e-9
        and float(da_net_trades["net_buy"].sum()) <= 1e-9
        and float(da_net_trades["net_sell"].sum()) <= 1e-9
    ):
        cap = 1.0  # must match cap used in build_battery_model
        free_mwh = max(0.0, cap - float(r_long_remaining_mwh))
        free_sell_budget_constr.RHS = float(free_mwh)
    else:
        # Tight "free" sell budget: only allow free sells backed by intraday-origin net-long.
        # Compute intraday-only signed net schedule (MW): (ALL trades) - (DA trades).
        signed_all = prev_net_trades["net_buy"].astype(float) - prev_net_trades["net_sell"].astype(float)
        signed_da = da_net_trades["net_buy"].astype(float) - da_net_trades["net_sell"].astype(float)
        signed_id = signed_all - signed_da

        # Free budget is intraday-origin net-long energy across the day (MWh).
        free_long_mwh = float(np.clip(signed_id.to_numpy(dtype=float), 0.0, None).sum() / 4.0)
        free_sell_budget_constr.RHS = max(0.0, free_long_mwh)

    # 3) Update netting RHS and bounds
    for i in T:
        prev_nb = float(prev_net_trades.loc[i, "net_buy"])
        prev_ns = float(prev_net_trades.loc[i, "net_sell"])
        prev_pos = prev_nb - prev_ns  # signed MW

        # RHS = carried net position into this bucket
        netting_constr[i].RHS = float(prev_pos)

        price = prices_qh.loc[i, "price"]
        if pd.isna(price):
            # Disable trading for missing price
            current_buy_qh[i].UB = 0.0
            current_sell_qh[i].UB = 0.0
            current_sell_free_qh[i].UB = 0.0
            current_sell_constr_qh[i].UB = 0.0
            continue

        # Allow trading in general
        current_buy_qh[i].UB = gp.GRB.INFINITY
        current_sell_qh[i].UB = gp.GRB.INFINITY

        # NEW
        # Constrained sells are only allowed if VWAP >= DA-buy reference price + tau
        # Otherwise we set UB=0 to hard-disable constrained sells for this product
        # Spread gate for constrained sells
        if (
            (p_da_buy is not None)
            and (float(r_long_remaining_mwh) > 1e-9)
            and (float(price) >= float(p_da_buy) + float(tau))
        ):
            current_sell_constr_qh[i].UB = gp.GRB.INFINITY
        else:
            current_sell_constr_qh[i].UB = 0.0

     
        # NEW
        # free sells are generally allowed (UB=infinity), but their total energy is controlled by FreeSellBudget (global constraint)
        current_sell_free_qh[i].UB = gp.GRB.INFINITY

    # 4) Rebuild objective function (unchanged, still uses current_sell_qh)
    obj = gp.LinExpr()
    for i in T:
        price = prices_qh.loc[i, "price"]
        if pd.isna(price):
            continue

        prev_nb = float(prev_net_trades.loc[i, "net_buy"])
        prev_ns = float(prev_net_trades.loc[i, "net_sell"])

        price_sell_adj = float(prices_qh_adj_all.loc[i, "price_sell_adj"])
        price_buy_adj = float(prices_qh_adj_all.loc[i, "price_buy_adj"])

        # Slightly different spread if there was no previous position
        if prev_nb < eps and prev_ns < eps:
            term = (
                current_sell_qh[i] * (price_sell_adj - 0.1 / 2 - eps)
                - current_buy_qh[i] * (price_buy_adj + 0.1 / 2 + eps)
            ) / 4.0
        else:
            term = (
                current_sell_qh[i] * (price_sell_adj - eps)
                - current_buy_qh[i] * (price_buy_adj + eps)
            ) / 4.0

        obj += term

    # NEW
    # If the profit objective is indifferent between allocating volume to free vs constrained sells,
    # add a tiny bonus to constrained sells so the solver prefers unwinding DA-long whenever the gate is open
    prefer_eps = 1e-4  # €/MWh, tiny tie-breaker
    obj += prefer_eps * gp.quicksum(current_sell_constr_qh[i] / 4.0 for i in T)

    m.setObjective(obj, gp.GRB.MAXIMIZE)

    # 5) Optimize
    m.optimize()

    if m.status != gp.GRB.OPTIMAL:
        logger.warning("No optimal solution found for current bucket.")
        empty_trades = pd.DataFrame(
            columns=["execution_time", "side", "quantity", "price", "product", "profit"]
        )
        return None, empty_trades, 0.0, float(r_long_remaining_mwh)

    # 6) Collect results
    results = pd.DataFrame(
        index=prices_qh.index,
        columns=[
            "current_buy_qh",
            "current_sell_qh",
            "current_sell_free_qh",
            "current_sell_constr_qh",
            "net_buy",
            "net_sell",
            "charge_sign",
            "battery_soc",
        ],
    )


    trade_rows = []
    for i in T:
        cb = float(current_buy_qh[i].X)
        cs = float(current_sell_qh[i].X)
        cs_free = float(current_sell_free_qh[i].X)
        cs_constr = float(current_sell_constr_qh[i].X)

        if cb > 0:
            trade_rows.append(
                (
                    execution_time,
                    "buy",
                    cb,
                    float(prices_qh.loc[i, "price"]),
                    i,
                    -cb * float(prices_qh.loc[i, "price"]) / 4.0,
                    cs_free,
                    cs_constr,
                )
            )
        if cs > 0:
            trade_rows.append(
                (
                    execution_time,
                    "sell",
                    cs,
                    float(prices_qh.loc[i, "price"]),
                    i,
                    cs * float(prices_qh.loc[i, "price"]) / 4.0,
                    cs_free,
                    cs_constr,
                )
            )

        results.loc[i, "current_buy_qh"] = cb
        results.loc[i, "current_sell_qh"] = cs
        results.loc[i, "current_sell_free_qh"] = cs_free
        results.loc[i, "current_sell_constr_qh"] = cs_constr
        results.loc[i, "net_buy"] = float(net_buy[i].X)
        results.loc[i, "net_sell"] = float(net_sell[i].X)
        results.loc[i, "charge_sign"] = float(charge_sign[i].X)
        results.loc[i, "battery_soc"] = float(vars["battery_soc"][i].X)

    trades = pd.DataFrame(
        trade_rows,
        columns=[
            "execution_time",
            "side",
            "quantity",
            "price",
            "product",
            "profit",
            "free_sell",
            "constr_sell",
        ],
    )

    # NEW
    # Update remaining DA-long (energy, MWh) by realized constrained sells in this bucket
    sold_constr_mwh = float(sum(current_sell_constr_qh[i].X / 4.0 for i in T))
    r_long_remaining_mwh_new = max(0.0, float(r_long_remaining_mwh) - sold_constr_mwh)

    return results, trades, float(m.ObjVal), float(r_long_remaining_mwh_new)





def derive_day_ahead_trades_from_drl_output(
    output: pd.DataFrame,
    current_day: pd.Timestamp,
) -> pd.DataFrame:
    """
    Convert DRL day-ahead output (hourly bids) into quarter-hourly trades.

    DRL output is expected to be indexed by date (string or date-like)
    and contain at least the columns:
        - capacity_trade
        - epex_spot_60min_de_lu_eur_per_mwh

    Returned columns:
        execution_time, side, quantity, price, product, profit
    """
    day_ahead_trades: Dict[pd.Timestamp, Dict[str, Any]] = {}

    # Extract the DRL row(s) for the current delivery day
    df = output.loc[current_day.date().isoformat()].copy()

    # Filter out zero trades
    mask = df.capacity_trade != 0
    df = df[mask]

    # Determine side and absolute volume
    df["side"] = ["buy" if x < 0 else "sell" for x in df.capacity_trade]
    df["net_volume"] = [abs(x) for x in df.capacity_trade]

    # Profit per hour
    df["profit"] = df.capacity_trade * df.epex_spot_60min_de_lu_eur_per_mwh

    # Move time from index to column
    df.reset_index(inplace=True)

    # Day-ahead clearing time: previous day at 13:00
    day_ahead_market_clearing = (current_day - pd.Timedelta(days=1)).replace(hour=13)

    for _, row in df.iterrows():
        # Split hourly position into 4 quarter-hours
        product_indexes = pd.date_range(row["time"], periods=4, freq="15min")

        for product_index in product_indexes:
            day_ahead_trades[product_index] = {
                "execution_time": day_ahead_market_clearing,
                "side": row["side"],
                "quantity": row["net_volume"],
                "price": row["epex_spot_60min_de_lu_eur_per_mwh"],
                "product": product_index,
                "profit": row["profit"] / 4,
            }

    return pd.DataFrame(day_ahead_trades).T.reset_index(drop=True)




def simulate_days_stacked_quarterhourly_products(
    da_bids_path: str,
    output_path: str,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    discount_rate: float,
    c_rate: float,
    roundtrip_eff: float,
    max_cycles: float,
    min_trades: float,  # kept for compatibility; currently unused
    vwaps_base_path: str = PRECOMPUTED_VWAP_PATH,
) -> None:
    """
    Test-mode simulation with CSV outputs

    - Reads DRL day-ahead bids from da_bids_path.
    - Writes:
        * trades per day -> {output_path}/trades/trades_YYYY-MM-DD.csv
        * VWAP logs       -> {output_path}/vwap/vwaps_YYYY-MM-DD.csv
        * profit/cycles   -> {output_path}/profit.csv
    """
    log_message = (
        "Running FAST rolling intrinsic QH TEST with parameters:\n"
        f"Start Day: {start_day}\n"
        f"End Day: {end_day}\n"
        f"Discount Rate: {discount_rate}\n"
        #f"Bucket Size: {bucket_size}\n"
        f"C Rate: {c_rate}\n"
        f"Roundtrip Efficiency: {roundtrip_eff}\n"
        f"Max Cycles: {max_cycles}\n"
        f"Min Trades: {min_trades}"
    )
    logger.info(log_message)

    tradepath = os.path.join(output_path, "trades")
    vwappath = os.path.join(output_path, "vwap")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tradepath, exist_ok=True)
    os.makedirs(vwappath, exist_ok=True)

    profitpath = os.path.join(output_path, "profit.csv")

    # Load or initialize profit/cycle history
    if os.path.exists(profitpath):
        profits = pd.read_csv(profitpath)
    else:
        profits = pd.DataFrame(columns=["day", "profit", "cycles"])

    if len(profits) > 0:
        current_day = (
            pd.Timestamp(profits.iloc[-1]["day"], tz="Europe/Berlin")
            + pd.Timedelta(days=1)
            + pd.Timedelta(hours=2)
        )
        current_cycles = float(profits.iloc[-1]["cycles"])
    else:
        current_day = start_day
        current_cycles = 0.0

    # DRL day-ahead bids
    drl_output = pd.read_csv(da_bids_path, index_col="time", parse_dates=True)
    drl_output.index = drl_output.index.tz_convert("Europe/Berlin")

    efficiency = roundtrip_eff ** 0.5  # for cycle tracking

    while current_day < end_day:
        current_day = current_day.replace(hour=0, minute=0, second=0, microsecond=0)
        logger.info(f"Current delivery day: {current_day}")

        all_trades = pd.DataFrame(
            columns=["execution_time", "side", "quantity", "price", "product", "profit", "market"]
        )

        # Derive day-ahead trades from DRL output
        try:
            day_ahead_trades_drl = derive_day_ahead_trades_from_drl_output(
                drl_output, current_day
            )
            day_ahead_trades_drl["market"] = "DA"
            all_trades = pd.concat(
                [all_trades, day_ahead_trades_drl], ignore_index=True
            )

        except KeyError:
            logger.warning(
                f"No DRL day-ahead trades for {current_day:%Y-%m-%d} in file "
                f"{da_bids_path} – intraday-only optimization."
            )

        trading_start = current_day - pd.Timedelta(hours=8)
        trading_end = current_day + pd.Timedelta(days=1)

        logger.info(f"Trading start: {trading_start}")
        logger.info(f"Trading end:   {trading_end}")

        # NEW
        # DA reference price (weighted avg buy price) and remaining DA-long (MWh)
        p_da_buy: float | None = None
        r_long_remaining_mwh: float = 0.0
        da_net_trades = get_net_trades(
            pd.DataFrame(columns=["execution_time", "side", "quantity", "price", "product", "profit"]),
            trading_end,
        )

        # NEW
        # Compute a DA reference buy price (weighted average) used for the constrained-sell spread gate
        # p_da_buy is None if there were no DA buy
        try:
            # Weighted average DA buy price (€/MWh)
            da_buys = day_ahead_trades_drl[day_ahead_trades_drl["side"] == "buy"]
            if len(da_buys) > 0:
                q = da_buys["quantity"].to_numpy(dtype=float)
                p = da_buys["price"].to_numpy(dtype=float)
                p_da_buy = float((q * p).sum() / max(1e-12, q.sum()))

            # DA-long surplus in energy terms (MWh): sum(max(net_pos,0))/4
            da_net_trades = get_net_trades(day_ahead_trades_drl, trading_end)
            buy_mwh = float(da_net_trades["net_buy"].sum()) / 4.0
            sell_mwh = float(da_net_trades["net_sell"].sum()) / 4.0
            r_long_remaining_mwh = max(0.0, buy_mwh * efficiency - sell_mwh / efficiency)

        except Exception as e:
            logger.warning(f"Could not initialize DA-long state: {e}")
            p_da_buy = None
            r_long_remaining_mwh = 0.0
            da_net_trades = get_net_trades(
                pd.DataFrame(columns=["execution_time", "side", "quantity", "price", "product", "profit"]),
                trading_end,
            )

        # Battery time index (delivery quarter-hours)
        start_of_day = trading_end - pd.Timedelta(hours=2)
        start_of_day = start_of_day.replace(hour=0, minute=0)
        end_of_day = start_of_day.replace(hour=23, minute=45)

        T = list(
            pd.date_range(
                start_of_day, end_of_day, freq="15min", tz="Europe/Berlin"
            )
        )

        # Build persistent Gurobi model
        m, vars_dict, netting_constr, max_cycles_constr, da_long_budget_constr, free_sell_budget_constr = build_battery_model(
            T, cap=1.0, c_rate=c_rate, roundtrip_eff=roundtrip_eff
            )


        # Load VWAP matrix for this day
        try:
            vwaps_day = load_vwaps_for_day(current_day, vwaps_base_path)
        except FileNotFoundError as e:
            logger.warning(
                f"No precomputed VWAPs for {current_day:%Y-%m-%d}: {e}. "
                "Skipping this day."
            )
            current_day = current_day + pd.Timedelta(days=1) + pd.Timedelta(hours=2)
            continue

        execution_time_start = trading_start

        bucket_size = infer_bucket_size_minutes(vwaps_day)
        
        execution_time_end = trading_start + pd.Timedelta(minutes=bucket_size)

        days_left = (end_day - current_day).days
        days_done = (current_day - start_day).days
        # Same somewhat ad-hoc logic as in legacy test script
        allowed_cycles = 1 + max(0, days_done - current_cycles)

        logger.info(f"Days left:        {days_left}")
        logger.info(f"Current cycles:   {current_cycles:.3f}")
        logger.info(f"Allowed cycles/d: {allowed_cycles:.3f}")

        # Intraday simulation over all buckets
        while execution_time_end < trading_end:
            vwap = get_vwap_from_precomputed(
                vwaps_day,
                execution_time_end=execution_time_end,
                end_date=trading_end,
            )

            # VWAP logging (per-bucket row) similar to old test script
            vwaps_for_logging = (
                vwap.copy().rename(columns={"price": execution_time_end}).T
            )
            vwap_filename = os.path.join(
                vwappath, f"vwaps_{current_day.strftime('%Y-%m-%d')}.csv"
            )

            if not os.path.exists(vwap_filename):
                vwaps_for_logging.to_csv(
                    vwap_filename,
                    mode="a",
                    header=True,
                    index=True,
                )
            elif os.path.exists(vwap_filename) and (
                execution_time_start == trading_start
            ):
                # First bucket of the day: overwrite previous file
                os.remove(vwap_filename)
                vwaps_for_logging.to_csv(
                    vwap_filename,
                    mode="a",
                    header=True,
                    index=True,
                )
            else:
                vwaps_for_logging.to_csv(
                    vwap_filename,
                    mode="a",
                    header=False,
                    index=True,
                )

            net_trades = get_net_trades(all_trades, trading_end)

            if vwap["price"].isnull().all():
                logger.info(
                    f"No VWAP prices in bucket "
                    f"[{execution_time_start}, {execution_time_end}) – skipping."
                )
                execution_time_start = execution_time_end
                execution_time_end = (
                    execution_time_start + pd.Timedelta(minutes=bucket_size)
                )
                continue
            
            tau = 1.0        
            try:
            # NEW
            # carry r_long across buckets
            # solve_bucket_with_persistent_model returns the updated r_long for the next bucket
                _, trades, profit, r_long_remaining_mwh= solve_bucket_with_persistent_model(
                        m=m,
                        vars=vars_dict,
                        netting_constr=netting_constr,
                        max_cycles_constr=max_cycles_constr,
                        da_long_budget_constr=da_long_budget_constr,
                        free_sell_budget_constr=free_sell_budget_constr,
                        prices_qh=vwap,
                        execution_time=execution_time_start,
                        discount_rate=discount_rate,
                        prev_net_trades=net_trades,
                        da_net_trades=da_net_trades,
                        allowed_cycles=allowed_cycles,
                        p_da_buy=p_da_buy,
                        tau=tau,
                        r_long_remaining_mwh=r_long_remaining_mwh,
                )

                # trades may be empty, but concat is robust
                trades["market"] = "IDC"
                all_trades = pd.concat(
                    [all_trades, trades], ignore_index=True
                )
            except ValueError as e:
                logger.error(
                    f"Error in optimization at execution_time_start="
                    f"{execution_time_start}: {e}"
                )

            execution_time_start = execution_time_end
            execution_time_end = (
                execution_time_start + pd.Timedelta(minutes=bucket_size)
            )

        # Daily profit
        daily_profit = all_trades["profit"].sum()

        # Net trades from final state -> cycle update
        net_trades = get_net_trades(all_trades, trading_end)
        current_cycles += net_trades["net_buy"].sum() / 4.0 * efficiency

        # Store trades
        trades_file = os.path.join(
            tradepath, f"trades_{current_day.strftime('%Y-%m-%d')}.csv"
        )
        all_trades.to_csv(trades_file, index=False)

        # Append profit/cycle history
        profits = pd.concat(
            [
                profits,
                pd.DataFrame(
                    [[current_day, daily_profit, current_cycles]],
                    columns=["day", "profit", "cycles"],
                ),
            ],
            ignore_index=True,
        )
        profits.to_csv(profitpath, index=False)

        logger.info(
            f"Finished day {current_day:%Y-%m-%d}: "
            f"profit={daily_profit:.2f}, cycles={current_cycles:.3f}"
        )

        # Next day
        current_day = current_day + pd.Timedelta(days=1) + pd.Timedelta(hours=2)
