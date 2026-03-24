# v3_counterfactual_reward_lookup

Same as **v2** (3 DA actions, counterfactual margins on first buy / first sell hour), but **intraday profit is not simulated during training**. It is read from precomputed CSVs:

- `coordinated_market_upper_bound_analysis/results/summary_YYYY-MM-DD.csv`  
  (or `PRECOMPUTED_DA_RI_SUMMARY_DIR` in `src/shared/config.py`)

Each file must contain columns including: `kind`, `buy_hour`, `sell_hour`, `profit`, `da_profit`.

## Hour mapping

Env timestep `0..23` → CSV `buy_hour` / `sell_hour` = **timestep + 1** (1..24), matching your batch generator.

## Behaviour

- **Full** episode: `profit` is looked up from the rollout **action** sequence (`kind` / `buy_hour` / `sell_hour`).
- **Counterfactual** at hour `t`: the env is **replayed** with idle at `t` (same as v2); the first buy/sell **realized volumes** define the schedule key, then `profit` is looked up. (Raw “idle substitution” on actions alone can be invalid, e.g. sell before buy.)
- Margin: `(profit_full - profit_cf) / RI_REWARD_SCALE` on the CF step (same structure as v2).
- If lookup fails: **0 EUR**, **`warnings.warn`**, TensorBoard **`lookup/miss_count_rollout`**.

**No live Gurobi RI** in the training loop → large speedup (replay is cheap vs full RI).

## Run

From repo root:

```bash
PYTHONPATH=. python -m src.coordinated_multi_market.v3_counterfactual_reward_lookup.train
```

Ensure every training day has a `summary_<date>.csv` (same dates as in `input_data`).
