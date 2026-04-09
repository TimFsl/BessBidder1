import os
import pandas as pd
from loguru import logger
from pathlib import Path
import numpy as np

from coordinated_market_upper_bound_analysis.da_trades_combinations import generate_da_combinations, create_synthetic_drl_output_for_combinations
from coordinated_market_upper_bound_analysis.testing_rolling_intrinsic_qh_intelligent_stacking_open_positions_V1 import (
    derive_day_ahead_trades_from_drl_output,
    load_vwaps_for_day,
    infer_bucket_size_minutes,
    get_vwap_from_precomputed,
    get_net_trades,
    build_battery_model,
    solve_bucket_with_persistent_model,
)

def run_ri_with_synthetic_da_trades(
    da_prices_path: str,
    output_path: str,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    discount_rate: float,
    c_rate: float,
    roundtrip_eff: float,
    max_cycles: float,
    vwaps_base_path: str,
    volume_mwh: float,
    tau: float = 1.0,
):
    os.makedirs(output_path, exist_ok=True)

    drl_output = pd.read_csv(da_prices_path, index_col="time", parse_dates=True)
    drl_output.index = drl_output.index.tz_convert("Europe/Berlin")

    combos = generate_da_combinations()
    efficiency = roundtrip_eff ** 0.5

    current_day = start_day

    while current_day < end_day:
        current_day = current_day.replace(hour=0, minute=0, second=0, microsecond=0)
        logger.info(f"Day: {current_day:%Y-%m-%d}")

        # Preise für den Tag müssen vorhanden sein:
        day_start = current_day
        day_hours = pd.date_range(day_start, periods=24, freq="1h", tz="Europe/Berlin")
        if not set(day_hours).issubset(set(drl_output.index)):
            logger.warning(f"Missing DA prices for {current_day:%Y-%m-%d}, skipping.")
            current_day = current_day + pd.Timedelta(days=1) + pd.Timedelta(hours=2)
            continue

        # VWAP laden
        try:
            vwaps_day = load_vwaps_for_day(current_day, vwaps_base_path)
        except FileNotFoundError as e:
            logger.warning(f"No VWAPs for {current_day:%Y-%m-%d}: {e}")
            current_day = current_day + pd.Timedelta(days=1) + pd.Timedelta(hours=2)
            continue

        bucket_size = infer_bucket_size_minutes(vwaps_day)

        # Setup RI horizon + Model
        trading_start = current_day - pd.Timedelta(hours=8)
        trading_end = current_day + pd.Timedelta(days=1)

        start_of_day = trading_end - pd.Timedelta(hours=2)
        start_of_day = start_of_day.replace(hour=0, minute=0)
        end_of_day = start_of_day.replace(hour=23, minute=45)

        T = list(pd.date_range(start_of_day, end_of_day, freq="15min", tz="Europe/Berlin"))
        
        #m, vars_dict, netting_constr, max_cycles_constr = build_battery_model(
        #    T, cap=1.0, c_rate=c_rate, roundtrip_eff=roundtrip_eff
        #)

        allowed_cycles = 1.0

        day_results = []
        for combo in combos:
            trade_cols = [
                "execution_time",
                "side",
                "quantity",
                "price",
                "product",
                "profit",
                "free_sell",
                "constr_sell",
                "r_long_before",
                "r_long_after",
                "free_mwh",
            ]
            all_trades = pd.DataFrame(columns=trade_cols)
            da_trades = pd.DataFrame(columns=trade_cols)

            p_da_buy: float | None = None
            r_long_remaining_mwh: float = 0.0
            da_net_trades = get_net_trades(pd.DataFrame(columns=["product", "side", "quantity"]), trading_end)

            if combo.kind != "no_da":
                synthetic_output = create_synthetic_drl_output_for_combinations(
                    drl_output,
                    current_day,
                    combo,
                    volume_mwh=volume_mwh,
                    roundtrip_eff=roundtrip_eff,
                )
                da_trades = derive_day_ahead_trades_from_drl_output(synthetic_output, current_day)
                da_trades["free_sell"] = 0.0
                da_trades["constr_sell"] = 0.0
                da_trades["r_long_before"] = r_long_remaining_mwh
                da_trades["r_long_after"] = r_long_remaining_mwh
                da_trades["free_mwh"] = 0.0
                all_trades = pd.concat([all_trades, da_trades], ignore_index=True)

            # init strictly from da_trades
            if not da_trades.empty:
                da_buys = da_trades[da_trades["side"] == "buy"]
                if len(da_buys) > 0:
                    q = da_buys["quantity"].to_numpy(dtype=float)
                    p = da_buys["price"].to_numpy(dtype=float)
                    p_da_buy = float((q * p).sum() / max(1e-12, q.sum()))

                da_net_trades = get_net_trades(da_trades, trading_end)
                #da_pos_mw = (da_net["net_buy"] - da_net["net_sell"]).to_numpy(dtype=float)
                #r_long_remaining_mwh = float(np.clip(da_pos_mw, 0.0, None).sum() / 4.0)
                
                #buy_mwh = max(0.0, float((da_net["net_buy"]).sum() / 4.0))
                #sell_mwh = max(0.0, float((da_net["net_sell"]).sum() / 4.0))
                #r_long_remaining_mwh = max(0.0, buy_mwh * roundtrip_eff - sell_mwh)
                #r_long_remaining_mwh = max(0.0, buy_mwh * efficiency - sell_mwh/efficiency)

                #r_long_remaining_mwh = max(0.0, float((da_net["net_buy"] - da_net["net_sell"]).sum() / 4.0))

                buy_mwh = float(da_net_trades["net_buy"].sum()) / 4.0
                sell_mwh = float(da_net_trades["net_sell"].sum()) / 4.0
                r_long_remaining_mwh = max(0.0, buy_mwh * efficiency - sell_mwh/efficiency)




            # Build Model
            m, vars_dict, netting_constr, max_cycles_constr, da_long_budget_constr, free_sell_budget_constr = build_battery_model(
                T, cap=1.0, c_rate=c_rate, roundtrip_eff=roundtrip_eff
            )

            allowed_cycles = 1.0

            execution_time_start = trading_start
            execution_time_end = trading_start + pd.Timedelta(minutes=bucket_size)

            while execution_time_end < trading_end:
                vwap = get_vwap_from_precomputed(vwaps_day, execution_time_end, trading_end)
                if vwap["price"].isnull().all():
                    execution_time_start = execution_time_end
                    execution_time_end = execution_time_start + pd.Timedelta(minutes=bucket_size)
                    continue

                net_trades = get_net_trades(all_trades, trading_end)


                # only for logging calculate free_mwh
                r_long_before = float(r_long_remaining_mwh)
                # Tight free budget (MWh): intraday-only net-long = (ALL net) - (DA net)
                signed_all = net_trades["net_buy"].astype(float) - net_trades["net_sell"].astype(float)
                signed_da = da_net_trades["net_buy"].astype(float) - da_net_trades["net_sell"].astype(float)
                signed_id = signed_all - signed_da
                free_mwh = float(np.clip(signed_id.to_numpy(dtype=float), 0.0, None).sum() / 4.0)




                _, trades, _, r_long_remaining_mwh = solve_bucket_with_persistent_model(
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

                trades["r_long_before"] = r_long_before
                trades["r_long_after"] = r_long_remaining_mwh
                trades["free_mwh"] = free_mwh

                all_trades = pd.concat([all_trades, trades], ignore_index=True)

                execution_time_start = execution_time_end
                execution_time_end = execution_time_start + pd.Timedelta(minutes=bucket_size)

            daily_profit = float(all_trades["profit"].sum())

            # Trades speichern
            day_dir = Path(output_path) / "trades" / f"{current_day:%Y-%m-%d}"
            day_dir.mkdir(parents=True, exist_ok=True)
            trades_file = day_dir / f"trades_combo_{combo.combo_id:03d}.csv"
            all_trades.to_csv(trades_file, index=False)

            # Summary-Zeile
            day_results.append({
                "day": current_day,
                "combo_id": combo.combo_id,
                "kind": combo.kind,
                "buy_hour": getattr(combo, "buy_hour", None),
                "sell_hour": getattr(combo, "sell_hour", None),
                "profit": daily_profit,
            })

        pd.DataFrame(day_results).to_csv(
            os.path.join(output_path, f"summary_{current_day:%Y-%m-%d}.csv"),
            index=False,
        )

        current_day = current_day + pd.Timedelta(days=1) + pd.Timedelta(hours=2)

if __name__ == "__main__":
    run_ri_with_synthetic_da_trades(
        da_prices_path="data/data_2019-01-01_2025-12-31_hourly.csv",
        output_path="coordinated_market_upper_bound_analysis/results_open_positions_V3/",
        start_day=pd.Timestamp("2024-12-31", tz="Europe/Berlin"),
        end_day=pd.Timestamp("2025-12-31", tz="Europe/Berlin"),
        discount_rate=0.0,
        c_rate=1.0,
        roundtrip_eff=0.86,
        max_cycles=999,
        vwaps_base_path="data/precomputed_vwaps",
        volume_mwh=1.0
    )
