import os
from datetime import datetime, timedelta
from dataclasses import dataclass

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import models
import utils
import simulation
from simulation import BacktestParams, BacktestColumns


@dataclass
class AnalysisParams:
    pass


def _get_closest_contract_symbol(
    target_date: datetime,
    symbol_to_expiration: dict[str, datetime],
    time_limit_days: int = 30,
) -> str:
    """Get the closest contract symbol based on expiration date."""
    time_diff = timedelta(days=time_limit_days)
    closest_date = None
    near_symbol = None
    for symbol, exp_date in symbol_to_expiration.items():
        if exp_date > target_date + time_diff:
            if closest_date is None or (exp_date - target_date) < (
                closest_date - target_date
            ):
                closest_date = exp_date
                near_symbol = symbol
    if near_symbol is None:
        raise ValueError("No suitable contract symbol found.")
    return near_symbol


def _increment_symbol(near_symbol: str) -> str:
    """Increment the contract by one year."""
    new = near_symbol[:-1] + str(int(near_symbol[-1]) + 1)
    return new


def _get_spread_symbol(near_symbol: str) -> str:
    new = near_symbol + str(int(near_symbol[-1]) + 1)
    return new


def _create_combined_dataframe(
    dir_path: str, qid_to_symbol: dict[str, str]
) -> pd.DataFrame:
    combined_data = {}
    for file_name in os.listdir(dir_path):
        date_str = file_name.split("_")[-1].split(".")[0]
        data = utils.read_data(
            file_name=file_name.split(".")[0],
            file_ext="parquet",
            dir_path=dir_path,
        )
        if data is not None:
            date = pd.to_datetime(date_str, format="%Y%m%d")
            if date not in combined_data:
                combined_data[date] = {}
            for _, row in data.iterrows():
                qid = row["qid"]
                symbol = row["settlement_price"]
                combined_data[date][qid] = symbol
    combined_df = pd.DataFrame.from_dict(combined_data, orient="index")
    combined_df = combined_df.rename(columns=qid_to_symbol)
    return combined_df


def _add_spread_prices(
    df: pd.DataFrame, return_ratio: bool = True
) -> pd.DataFrame:
    suffixes = ["2", "3", "4", "5", "6"]
    letters = ["H", "K", "N", "U", "Z"]
    for letter in letters:
        for i in range(1, len(suffixes)):
            near_suffix = suffixes[i - 1]
            far_suffix = suffixes[i]
            for prefix in df.columns:
                if prefix.endswith(letter + near_suffix):
                    base = prefix[:-2]
                    near_col = base + letter + near_suffix
                    far_col = base + letter + far_suffix
                    if near_col in df.columns and far_col in df.columns:
                        new_col_name = base + letter + near_suffix + far_suffix
                        if return_ratio:
                            df[new_col_name] = df[near_col] / df[far_col]
                        else:
                            df[new_col_name] = df[near_col] - df[far_col]
    return df


def _add_continuous_spread_price(
    data: pd.DataFrame,
    symbol_to_expiration: dict[str, datetime],
    direction: str = "backward",
) -> pd.DataFrame:
    """Create a continuous time series from separate futures time series."""
    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be either 'forward' or 'backward'")

    last_spread = None
    adjustment = 0
    dates = data.index if direction == "backward" else reversed(data.index)

    continuous_prices = np.zeros(len(data))
    total_adjustment = np.zeros(len(data))
    near_series = np.empty(len(data), dtype=object)
    far_series = np.empty(len(data), dtype=object)
    spread_series = np.empty(len(data), dtype=object)
    near_contract_prices = np.zeros(len(data))
    far_contract_prices = np.zeros(len(data))
    preroll_near_contract_prices = np.zeros(len(data))
    preroll_far_contract_prices = np.zeros(len(data))
    rollovers = np.zeros(len(data), dtype=bool)

    last_near_symbol = None
    last_far_symbol = None

    for i, date in enumerate(dates):
        near_symbol = _get_closest_contract_symbol(date, symbol_to_expiration)
        far_symbol = _increment_symbol(near_symbol)
        spread_symbol = _get_spread_symbol(near_symbol)

        if last_spread is not None and spread_symbol != last_spread:
            adjustment += (
                data.at[date, spread_symbol] - data.at[date, last_spread]
            )
            rollovers[i] = True
        else:
            rollovers[i] = False

        continuous_prices[i] = np.log(
            data.at[date, spread_symbol] - adjustment
        )
        near_series[i] = near_symbol
        far_series[i] = far_symbol
        spread_series[i] = spread_symbol
        near_contract_prices[i] = data.at[date, near_symbol]
        far_contract_prices[i] = (
            data.at[date, far_symbol] if far_symbol in data.columns else np.nan
        )

        total_adjustment[i] = adjustment
        last_spread = spread_symbol

        if last_near_symbol is not None and last_far_symbol is not None:
            preroll_near_contract_prices[i] = data.at[date, last_near_symbol]
            preroll_far_contract_prices[i] = data.at[date, last_far_symbol]
        else:
            preroll_near_contract_prices[i] = np.nan
            preroll_far_contract_prices[i] = np.nan

        last_near_symbol = near_symbol
        last_far_symbol = far_symbol

    result_df = pd.DataFrame(
        {
            "near_contract_symbol": near_series,
            "far_contract_symbol": far_series,
            "spread_symbol": spread_series,
            "near_contract_price": near_contract_prices,
            "far_contract_price": far_contract_prices,
            "preroll_near_contract_price": preroll_near_contract_prices,
            "preroll_far_contract_price": preroll_far_contract_prices,
            "continuous_price": continuous_prices,
            "spread": near_contract_prices - far_contract_prices,
            "price_adjustment": total_adjustment,
            "rollover": rollovers,
        },
        index=data.index,
    )

    data = data.join(result_df)

    return data


def _add_model_parameter_fit(
    data: pd.DataFrame, column_name: str, window_size: int = 100
) -> pd.DataFrame:
    """Fit Ornstein-Uhlenbeck parameters to data within a rolling window."""
    array = utils.convert_to_array(data[column_name])
    windows = sliding_window_view(array, window_shape=window_size)

    mu_values = np.zeros(len(windows))
    kappa_values = np.zeros(len(windows))
    var_values = np.zeros(len(windows))
    var_stationary_values = np.zeros(len(windows))
    alpha_values = np.zeros(len(windows))
    beta_values = np.zeros(len(windows))

    for i, window in enumerate(iterable=windows):
        ou_params = models.fit_ou_process(window)
        mu_values[i] = ou_params.mu
        kappa_values[i] = ou_params.kappa
        var_values[i] = ou_params.var
        var_stationary_values[i] = ou_params.var_stationary
        alpha_values[i] = ou_params.alpha
        beta_values[i] = ou_params.beta

    params_array = np.column_stack(
        (
            mu_values,
            kappa_values,
            var_values,
            var_stationary_values,
            alpha_values,
            beta_values,
        )
    )

    params_df = pd.DataFrame(
        data=params_array,
        columns=["mu", "kappa", "var", "var_stationary", "alpha", "beta"],
        index=data.index[window_size - 1 :],
    )

    data = data.join(other=params_df, how="left")

    return data


def _add_trading_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Add trading signal values to the DataFrame based on the OU score."""
    continuous_price = utils.convert_to_array((data["continuous_price"]))
    mu = utils.convert_to_array(data["mu"])
    var_stationary = utils.convert_to_array(data["var_stationary"])
    mr_signal = (continuous_price - mu) / np.sqrt(var_stationary)
    data["mr_signal"] = mr_signal
    return data


def _visualize_strategy(
    results_df: pd.DataFrame,
    params: BacktestParams,
    subplot: int | None = None,
    n: int | None = None,
) -> None:
    """Visualize strategy performance based on the results DataFrame."""

    results_df["lower_entry_threshold"] = params.lower_entry_threshold
    results_df["upper_entry_threshold"] = params.upper_entry_threshold
    results_df["lower_exit_threshold"] = params.lower_exit_threshold
    results_df["upper_exit_threshold"] = params.upper_exit_threshold

    position = results_df["position"]

    results_df["enter_long"] = (position.shift(1) == 0) & (position == 1)
    results_df["exit_long"] = (position.shift(1) == 1) & (
        position.isin([0, -1])
    )
    results_df["enter_short"] = (position.shift(1) == 0) & (position == -1)
    results_df["exit_short"] = (position.shift(1) == -1) & (
        position.isin([0, 1])
    )

    plots = [
        lambda: utils.plot_data(
            data=results_df[
                [
                    "mr_signal",
                    "lower_entry_threshold",
                    "lower_exit_threshold",
                    "upper_entry_threshold",
                    "upper_exit_threshold",
                    "enter_long",
                    "exit_long",
                    "enter_short",
                    "exit_short",
                    "rollover",
                ]
            ],
            annotation_columns=[
                "rollover",
                "enter_long",
                "exit_long",
                "enter_short",
                "exit_short",
            ],
            hidden_columns=[
                "lower_exit_threshold",
                "upper_exit_threshold",
                "enter_long",
                "exit_long",
                "enter_short",
                "exit_short",
                "rollover",
            ],
        ),
        lambda: utils.plot_data(
            data=results_df[
                [
                    "continuous_price",
                    "mu",
                    "enter_long",
                    "exit_long",
                    "enter_short",
                    "exit_short",
                    "rollover",
                ]
            ],
            annotation_columns=[
                "enter_long",
                "exit_long",
                "enter_short",
                "exit_short",
                "rollover",
            ],
            hidden_columns=[
                "mu",
                "rollover",
                "enter_short",
                "exit_short",
                "exit_long",
            ],
        ),
        lambda: utils.plot_data(
            data=results_df[
                [
                    "spread",
                    "near_contract_price",
                    "far_contract_price",
                    "enter_long",
                    "exit_long",
                    "enter_short",
                    "exit_short",
                    "rollover",
                ]
            ],
            annotation_columns=[
                "enter_long",
                "exit_long",
                "enter_short",
                "exit_short",
                "rollover",
            ],
            hidden_columns=["rollover", "exit_short", "enter_short", "exit_long"],
        ),
        lambda: utils.plot_data(
            data=results_df[
                ["pnl", "enter_long", "exit_long", "enter_short", "exit_short"]
            ],
            annotation_columns=[
                "enter_long",
                "exit_long",
                "enter_short",
                "exit_short",
            ],
            hidden_columns=["exit_short", "enter_short", "exit_long"],
        ),
    ]

    if n is None:
        for plot in plots:
            plot()
    else:
        if 1 <= n <= len(plots):
            plots[n - 1]()
        else:
            print(f"Invalid input: {n}. Please enter a value between 1 and {len(plots)}.")


def main_analysis() -> list[pd.DataFrame]:
    """Main analysis workflow."""
    dir_path = "data/future_chain_time_series"
    qid_map = utils.read_data(
        dir_path="data", file_name="map_qid_to_future_contract", file_ext="csv"
    )
    qid_map["expiration_date"] = qid_map["expiration_date"].apply(
        utils.convert_to_datetime
    )
    qid_to_symbol = qid_map.set_index("qid")["symbol"].to_dict()
    symbol_to_expiration = qid_map.set_index("symbol")[
        "expiration_date"
    ].to_dict()

    combined_df = _create_combined_dataframe(dir_path, qid_to_symbol)
    ratios_df = _add_spread_prices(combined_df)
    continuous_df = _add_continuous_spread_price(
        ratios_df, symbol_to_expiration
    )
    fitted_df = _add_model_parameter_fit(
        continuous_df, column_name="continuous_price"
    )
    signals_df = _add_trading_signals(fitted_df)
    split_index = int(len(signals_df) * 0.8)
    training_df = signals_df.iloc[:split_index]
    testing_df = signals_df.iloc[split_index:]

    cols = BacktestColumns(
        signal_column="mr_signal",
        constraint_column="kappa",
        p1_price_column="near_contract_price",
        r1_price_column="preroll_near_contract_price",
        p2_price_column="far_contract_price",
        r2_price_column="preroll_far_contract_price",
        rollover_column="rollover",
    )

    params = BacktestParams(
        lower_constraint=1,
        upper_constraint=100,
        lower_entry_threshold=-10,
        lower_exit_threshold=-4,
        upper_entry_threshold=10,
        upper_exit_threshold=4,
        contract_size=10,
        target_gross_exposure=1000000,
    )

    results_df = simulation.backtest(data=signals_df, cols=cols, params=params)

    _visualize_strategy(results_df, params)

    return [training_df, testing_df, results_df]


if __name__ == "__main__":
    main_analysis()
