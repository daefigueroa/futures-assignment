import math
from dataclasses import dataclass
from typing import Protocol, SupportsFloat, Callable
from functools import partial

import numpy as np
import numpy.typing as npt

from utils import convert_to_array


@dataclass
class StatsParameters:
    mean: float
    var: float
    median: float


@dataclass
class RegressionParameters:
    alpha: float
    beta: float
    var: float


@dataclass
class OUParameters:
    mu: float
    kappa: float
    var: float
    var_stationary: float
    alpha: float
    beta: float


@dataclass
class StrategyMetrics:
    sharpe_ratio: float
    calmar_ratio: float
    hit_ratio: float
    max_drawdown: int


@dataclass
class NormalDistParams:
    mu: float
    sigma: float


class NormalDistFunction(Protocol):
    def __call__(
        self, x: SupportsFloat, params: NormalDistParams
    ) -> float: ...


def ks_distance(
    empirical_data: npt.ArrayLike,
    comparison_dist: Callable[[SupportsFloat], float],
) -> float:
    """
    Calculate the Kolmogorov-Smirnov (KS) distance between dist functions.

    Args:
        empirical_data (npt.ArrayLike): Empirical data series.
        comparison_dist (Callable[[SupportsFloat], float]): Distribution
        function to compare against.

    Returns:
        float: KS distance.
    """
    empirical_data = convert_to_array(empirical_data)
    running_max = 0.0
    counter = 0
    n = len(empirical_data)
    for a in np.sort(empirical_data):
        left_diff = abs(comparison_dist(a) - counter / n)
        right_diff = abs(comparison_dist(a) - (counter + 1) / n)
        running_max = max(left_diff, right_diff, running_max)
        counter += 1
    return running_max


def generate_ks_distance_samples(
    n: int,
    m: int,
    params: NormalDistParams,
    sample_generator: Callable[[int, float, float], npt.NDArray[np.float64]],
    dist_function: NormalDistFunction,
) -> npt.NDArray[np.float64]:
    """
    Generate samples of KS distances using bootstrapping.

    Args:
        n (int): Number of samples per bootstrap.
        m (int): Number of bootstrap iterations.
        params (NormalDistParams): Parameters of the normal distribution.
        sample_generator (Callable[[int, float, float], npt.NDArray]):
            Function to generate samples.
        dist_function (NormalDistFunction): Normal distribution function.

    Returns:
        npt.NDArray[np.float64]: KS distance samples.
    """
    ks_distance_samples = []
    for _ in range(m):
        bootstrap_sample = sample_generator(n, params.mu, params.sigma)
        sample_statistics = calculate_statistics(bootstrap_sample)
        sample_params = NormalDistParams(
            mu=sample_statistics.mean, sigma=math.sqrt(sample_statistics.var)
        )
        bootstrapped_ks_distance = ks_distance(
            empirical_data=bootstrap_sample,
            comparison_dist=partial(dist_function, params=sample_params),
        )
        ks_distance_samples.append(bootstrapped_ks_distance)
    return np.array(ks_distance_samples)


def calculate_normal_parameters(
    mean: SupportsFloat, std_dev: SupportsFloat
) -> NormalDistParams:
    mean = float(mean)
    std_dev = float(std_dev)
    return NormalDistParams(mu=mean, sigma=std_dev)


def normal_dist_func(x: SupportsFloat, params: NormalDistParams) -> float:
    """
    Calculate the cumulative distribution function (CDF) of a normal
    distribution.

    Args:
        x (SupportsFloat): Value at which to evaluate the CDF.
        params (NormalDistParams): Parameters of the normal distribution.

    Returns:
        float: CDF value.
    """
    x = float(x)
    value = 0.5 * (
        1 + math.erf((x - params.mu) / (params.sigma * math.sqrt(2)))
    )
    return value


def calculate_statistics(series: npt.ArrayLike) -> StatsParameters:
    """
    Calculate statistical parameters for a given data series.

    Args:
        series (npt.ArrayLike): Input data series.

    Returns:
        StatsParameters: Statistical parameters such as mean, variance,
        and median.
    """
    array = np.array(object=series, dtype=np.float64)
    return StatsParameters(
        mean=float(array.mean()),
        var=float(array.var(ddof=1)),
        median=float(np.median(array)),
    )


def calculate_strategy_metrics(
    pnl_series: npt.ArrayLike, extrapolation_ratio: SupportsFloat
) -> StrategyMetrics:
    pnl_series = convert_to_array(pnl_series)
    extrapolation_ratio = float(extrapolation_ratio)
    stats = calculate_statistics(np.diff(pnl_series))

    sharpe_ratio = float((stats.mean / np.sqrt(stats.var)) * np.sqrt(
        extrapolation_ratio
    ))

    cumulative_returns = np.cumsum(pnl_series)
    drawdowns = cumulative_returns - np.maximum.accumulate(cumulative_returns)
    max_drawdown = int(np.min(drawdowns))

    calmar_ratio = float(stats.mean * extrapolation_ratio / abs(max_drawdown))

    hit_ratio = float(np.mean(np.diff(pnl_series) > 0))

    return StrategyMetrics(
        sharpe_ratio=sharpe_ratio,
        calmar_ratio=calmar_ratio,
        hit_ratio=hit_ratio,
        max_drawdown=max_drawdown,
    )


def fit_linear_regression(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
) -> RegressionParameters:
    """
    Fit a simple linear regression model to the given data.

    Args:
        x (npt.ArrayLike): Independent data.
        y (npt.ArrayLike): Dependent data.

    Returns:
        RegressionParameters: Parameters of the fitted linear regression
        model.
    """
    x = convert_to_array(x)
    y = convert_to_array(y)

    n = len(x)
    if n != len(y):
        raise ValueError("Input arrays are not of equal length.")
    if n < 3:
        raise ValueError("Not enough data to calculate estimates. (n < 3)")

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_squared = np.sum(x**2)

    beta = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
    alpha = np.mean(y) - beta * np.mean(x)

    residuals = y - (alpha + beta * x)
    var = np.sum(residuals**2) / (n - 2)

    return RegressionParameters(
        alpha=float(alpha), beta=float(beta), var=float(var)
    )


def calculate_residuals(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    params: RegressionParameters,
) -> npt.NDArray[np.float64]:
    """
    Calculate residuals given parameters of a linear regression model.

    Args:
        x (npt.ArrayLike): Independent data.
        y (npt.ArrayLike): Dependent data.
        params (RegressionParameters): Parameters of the linear regression
        model.

    Returns:
        npt.NDArray[np.float64]: Residuals of the linear regression model.
    """
    x = convert_to_array(x)
    y = convert_to_array(y)

    residuals = y - (params.alpha + params.beta * x)
    return residuals


def fit_ou_process(
    process_values: npt.ArrayLike, yearly_factor: SupportsFloat = 252
) -> OUParameters:
    """
    Fit an Ornstein-Uhlenbeck (OU) process to the given data.

    Args:
        process_values (npt.ArrayLike): Time series data of the process.
        yearly_factor (SupportsFloat): Factor to annualize the parameters.

    Returns:
        OUParameters: Parameters of the fitted OU process.
    """
    yearly_factor = float(yearly_factor)
    process_values = convert_to_array(process_values)
    X = process_values[:-1]
    Y = process_values[1:]
    r = fit_linear_regression(X, Y)

    kappa = -math.log(r.beta) * yearly_factor
    mu = r.alpha / (1 - r.beta)
    var = (r.var * 2 * kappa) / (1 - r.beta**2)
    var_stationary = r.var / (1 - r.beta**2)

    if var_stationary <= 0:
        var_stationary = np.nan

    return OUParameters(
        mu=mu,
        kappa=kappa,
        var=var,
        var_stationary=var_stationary,
        alpha=r.alpha,
        beta=r.beta,
    )
