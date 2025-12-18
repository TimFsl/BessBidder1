import warnings
import pandas as pd

from dotenv import load_dotenv

from src.shared.config import (
    C_RATE,
    MAX_CYCLES_PER_YEAR,
    MIN_TRADES,
    RTE,
    VAL_START,
    VAL_END,
)
#from src.single_market.rolling_intrinsic_new import simulate_period
#from src.single_market.rolling_intrinsic_new_copy import simulate_period
from src.single_market.rolling_intrinsic_gurobi_qh import simulate_period

load_dotenv()

warnings.simplefilter(action="ignore", category=FutureWarning)

if __name__ == "__main__":

    simulate_period(
        VAL_START,
        VAL_END,
        discount_rate=0,
        c_rate=C_RATE,
        roundtrip_eff=RTE,
        max_cycles=MAX_CYCLES_PER_YEAR,
        min_trades=MIN_TRADES,
    )
