import math
from typing import Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass()
class BacktestColumns:
    signal_column: str
    constraint_column: str
    p1_price_column: str
    p2_price_column: str
    r1_price_column: Optional[str] = None
    r2_price_column: Optional[str] = None
    rollover_column: Optional[str] = None


@dataclass
class BacktestParams:
    upper_constraint: float
    lower_constraint: float
    lower_entry_threshold: float
    lower_exit_threshold: float
    upper_entry_threshold: float
    upper_exit_threshold: float
    contract_size: int = 10
    target_gross_exposure: float = 1000000


def _determine_holdings_after_settlement(
    target_gross_exposure: float,
    p1_price: float,
    p2_price: float,
    contract_size: int,
    position_12_l: int,
) -> tuple[int, int]:
    """Calculate hedged holdings after settlement based on target exposure."""
    if position_12_l == 0:
        p1_held_as = 0
        p2_held_as = 0
    else:
        p1_held_as = math.ceil(
            (target_gross_exposure / 2)
            / (p1_price * contract_size)
            * position_12_l
        )
        p2_held_as = (
            math.ceil((target_gross_exposure / 2) / (p2_price * contract_size))
            * -position_12_l
        )
    return p1_held_as, p2_held_as


def backtest(
    data: pd.DataFrame, cols: BacktestColumns, params: BacktestParams
) -> pd.DataFrame:
    """
    Simulates a trading strategy on historical data.
    Returns:
    -------
    pd.DataFrame
        DataFrame containing the results of the backtest, including positions,
        holdings, exposures, PnL, and overshoot of target exposure.
    """
    position_list = []
    p1_held_bs_list = []
    r1_held_bs_list = []
    p2_held_bs_list = []
    r2_held_bs_list = []
    p1_held_as_list = []
    r1_held_as_list = []
    p2_held_as_list = []
    r2_held_as_list = []
    cash_held_as_list = []
    p1_exposure_list = []
    p2_exposure_list = []
    net_exposure_list = []
    gross_exposure_list = []
    gross_exposure_overshoot_list = []

    # Track values before and after settlement (bs/as)
    p1_held_bs = 0
    r1_held_bs = 0
    p2_held_bs = 0
    r2_held_bs = 0
    p1_held_as = 0
    r1_held_as = 0
    p2_held_as = 0
    r2_held_as = 0
    cash_held_bs = 0
    position_12_l = 0
    p1_price_prev = 0
    p2_price_prev = 0
    r1_price_prev = 0
    r2_price_prev = 0

    apply_rollover_logic = all(
        [
            cols.r1_price_column,
            cols.r2_price_column,
            cols.rollover_column,
        ]
    )

    dates = data.index
    for i, date in enumerate(dates):
        signal = data.at[date, cols.signal_column]
        constraint = data.at[date, cols.constraint_column]
        p1_price = data.at[date, cols.p1_price_column]
        p2_price = data.at[date, cols.p2_price_column]
        r1_price = data.at[date, cols.r1_price_column]
        r2_price = data.at[date, cols.r2_price_column]
        rollover = data.at[date, cols.rollover_column]

        if apply_rollover_logic and rollover:
            pnl = (
                r1_held_bs * (r1_price - r1_price_prev)
                + r2_held_bs * (r2_price - r2_price_prev)
            )
        else:
            pnl = (
                p1_held_bs * (p1_price - p1_price_prev)
                + p2_held_bs * (p2_price - p2_price_prev)
            )
        cash_held_as = cash_held_bs + pnl

        if position_12_l == 0:
            if params.lower_constraint < constraint < params.upper_constraint:
                if signal < params.lower_entry_threshold:
                    position_12_l = 1  # long product1, short product2
                elif signal > params.upper_entry_threshold:
                    position_12_l = -1  # short product1, long product2

        if (position_12_l == 1 and signal > params.lower_exit_threshold) or (
            position_12_l == -1 and signal < params.upper_exit_threshold
        ):
            position_12_l = 0

        p1_held_as, p2_held_as = _determine_holdings_after_settlement(
            params.target_gross_exposure,
            p1_price,
            p2_price,
            params.contract_size,
            position_12_l,
        )

        p1_exposure = p1_held_as * params.contract_size * p1_price
        p2_exposure = p2_held_as * params.contract_size * p2_price
        net_exposure = p1_exposure + p2_exposure
        gross_exposure = abs(p1_exposure) + abs(p2_exposure)
        gross_exposure_overshoot = (
            params.target_gross_exposure - gross_exposure
        )

        position_list.append(position_12_l)
        p1_held_bs_list.append(p1_held_bs)
        r1_held_bs_list.append(r1_held_bs)
        p2_held_bs_list.append(p2_held_bs)
        r2_held_bs_list.append(r2_held_bs)
        p1_held_as_list.append(p1_held_as)
        r1_held_as_list.append(r1_held_as)
        p2_held_as_list.append(p2_held_as)
        r2_held_as_list.append(r2_held_as)
        cash_held_as_list.append(cash_held_as)
        p1_exposure_list.append(p1_exposure)
        p2_exposure_list.append(p2_exposure)
        net_exposure_list.append(net_exposure)
        gross_exposure_list.append(gross_exposure)
        gross_exposure_overshoot_list.append(gross_exposure_overshoot)

        p1_price_prev = p1_price
        p2_price_prev = p2_price
        r1_price_prev = r1_price
        r2_price_prev = r2_price

        p1_held_bs = p1_held_as
        r1_held_bs = r1_held_as
        p2_held_bs = p2_held_as
        r2_held_bs = r2_held_as
        cash_held_bs = cash_held_as

    result_df = (
        pd.DataFrame(
            {
                "date": dates,
                "position": position_list,
                "p1_held_as": p1_held_as_list,
                "r1_held_as": r1_held_as_list,
                "p2_held_as": p2_held_as_list,
                "r2_held_as": r2_held_as_list,
                "pnl": cash_held_as_list,
                "p1_exposure": p1_exposure_list,
                "p2_exposure": p2_exposure_list,
                "net_exposure": net_exposure_list,
                "gross_exposure": gross_exposure_list,
                "gross_exposure_overshoot": gross_exposure_overshoot_list,
            }
        )
        .set_index("date")
        .join(data)
    )

    return result_df


def test_signal_sensitivities(
    data: pd.DataFrame,
    cols: BacktestColumns,
    lower_constraint: float,
    upper_constraint: float,
    lower_entry_th_min: float = -1,
    lower_entry_th_max: float = -5,
    upper_entry_th_min: float = 1,
    upper_entry_th_max: float = 10,
    upper_entry_exit_ratio: float = 0.5,
    lower_entry_exit_ratio: float = 0.5,
    n: int = 5,
    contract_size: int = 10,
    target_gross_exposure: float = 1000000
) -> pd.DataFrame:
    lower_entry_thresholds = np.linspace(
        lower_entry_th_min, lower_entry_th_max, n
    )
    upper_entry_thresholds = np.linspace(
        upper_entry_th_min, upper_entry_th_max, n
    )

    results = []

    for lower_entry in lower_entry_thresholds:
        for upper_entry in upper_entry_thresholds:
            lower_exit = lower_entry * lower_entry_exit_ratio
            upper_exit = upper_entry * upper_entry_exit_ratio

            params = BacktestParams(
                upper_constraint=upper_constraint,
                lower_constraint=lower_constraint,
                lower_entry_threshold=lower_entry,
                lower_exit_threshold=lower_exit,
                upper_entry_threshold=upper_entry,
                upper_exit_threshold=upper_exit,
                contract_size=contract_size,
                target_gross_exposure=target_gross_exposure
            )

            result_df = backtest(
                data=data,
                cols=cols,
                params=params
            )

            final_pnl = result_df["pnl"].iloc[-1]
            results.append(
                {
                    "lower_entry_threshold": lower_entry,
                    "upper_entry_threshold": upper_entry,
                    "final_pnl": final_pnl,
                }
            )

    results_df = pd.DataFrame(results)
    return results_df
