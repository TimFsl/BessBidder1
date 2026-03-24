# da_oracle_combos.py
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd


# da_trades_combinations.py
from dataclasses import dataclass
from typing import Optional, List

@dataclass(frozen=True)
class DACombo:
    combo_id: int
    kind: str  # "buy_only" | "buy_sell"
    buy_hour: Optional[int] = None
    sell_hour: Optional[int] = None


def generate_da_combinations() -> List[DACombo]:
    combos: List[DACombo] = []
    cid = 0

    combos.append(DACombo(combo_id=0, kind="no_da"))
    cid = 1

    # buy only (1..23)
    for b in range(1, 24):
        combos.append(DACombo(cid, "buy_only", buy_hour=b))
        cid += 1

    # buy -> sell (1<=b<s<=24)
    for b in range(1, 25):
        for s in range(b + 1, 25):
            combos.append(DACombo(cid, "buy_sell", buy_hour=b, sell_hour=s))
            cid += 1

    assert cid == 300, f"Expected 300, got {cid}"
    return combos

"""
#Simplified version: only buy-only combinations + no_da
def generate_da_combinations() -> List[DACombo]:
    combos: List[DACombo] = []
    cid = 0

    combos.append(DACombo(combo_id=0, kind="no_da"))
    cid = 1

    # buy only (1..23)
    for b in range(1, 24):
        combos.append(DACombo(cid, "buy_only", buy_hour=b))
        cid += 1

    assert cid == 24, f"Expected 24, got {cid}"
    return combos
"""

def create_synthetic_drl_output_for_combinations(
    da_price_frame: pd.DataFrame,
    current_day: pd.Timestamp,
    combo: DACombo,
    volume_mwh: float,
    roundtrip_eff: float,
) -> pd.DataFrame:
    day_start = current_day.replace(hour=0, minute=0, second=0, microsecond=0)
    hours = pd.date_range(day_start, periods=24, freq="1h", tz="Europe/Berlin")

    prices = da_price_frame.loc[hours, "epex_spot_60min_de_lu_eur_per_mwh"].astype(float)

    df = pd.DataFrame(index=hours)
    df.index.name = "time"
    df["epex_spot_60min_de_lu_eur_per_mwh"] = prices.values
    df["capacity_trade"] = 0.0

    def ts_of_hour(h: int) -> pd.Timestamp:
        return day_start + pd.Timedelta(hours=h - 1)

    def set_buy(h: int):
        df.loc[ts_of_hour(h), "capacity_trade"] = -volume_mwh #* (1/efficiency)  # buy (charge)

    def set_sell(h: int):
        df.loc[ts_of_hour(h), "capacity_trade"] = +volume_mwh * roundtrip_eff  # sell (discharge, losses)

    if combo.kind == "buy_only":
        set_buy(combo.buy_hour)
    elif combo.kind == "buy_sell":
        set_buy(combo.buy_hour)
        set_sell(combo.sell_hour)
    else:
        raise ValueError(combo.kind)

    return df
