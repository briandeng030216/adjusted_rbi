"""Core Win Expectancy, ARBI, and CRBI calculations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


GameCondition = tuple[int, float, int]
BaseState = tuple[int, int, int]
WinExpectancyFunction = Callable[[float | np.ndarray], np.ndarray]

BASE_STATE_LABELS = {
    "Empty": (0, 0, 0),
    "1B only": (1, 0, 0),
    "2B only": (0, 1, 0),
    "1B & 2B": (1, 1, 0),
    "3B only": (0, 0, 1),
    "1B & 3B": (1, 0, 1),
    "2B & 3B": (0, 1, 1),
    "Loaded": (1, 1, 1),
}


def read_win_expectancy_table(path: str | Path) -> pd.DataFrame:
    """Read the tab-delimited Win Expectancy reference table."""
    rows: list[list[object]] = []
    score_differences = list(range(-5, 6))

    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            values = line.strip().split("\t")
            if not values or values[0] == "Inn":
                continue

            inning = int(values[0])
            half_inning = 0.0 if values[1] == "Top" else 0.5
            outs = int(values[2])
            base_state = BASE_STATE_LABELS[values[3]]
            probabilities = [float(value) for value in values[4:]]
            probabilities.extend([-1.0] * (len(score_differences) - len(probabilities)))
            rows.append(
                [(inning, half_inning, outs), base_state, *probabilities[:11]]
            )

    return pd.DataFrame(
        rows,
        columns=["condition", "base_runners", *score_differences],
    )


def build_win_expectancy_splines(
    table: pd.DataFrame,
) -> dict[GameCondition, dict[BaseState, WinExpectancyFunction]]:
    """Extend the empirical WE grid with PCHIP and exponential tails."""
    splines: dict[GameCondition, dict[BaseState, WinExpectancyFunction]] = {}
    score_differences = np.arange(-5, 6)

    for _, row in table.iterrows():
        values = np.array([row[value] for value in score_differences], dtype=float)
        valid = values >= 0
        x_valid = score_differences[valid]
        y_valid = values[valid]
        raw_spline = PchipInterpolator(x_valid, y_valid, extrapolate=True)

        x_left, x_right = x_valid[0], x_valid[-1]
        value_left, value_right = y_valid[0], y_valid[-1]
        derivative_left = float(raw_spline.derivative()(x_left))
        derivative_right = float(raw_spline.derivative()(x_right))

        if derivative_left <= 0:
            derivative_left = max(
                (y_valid[1] - y_valid[0]) / (x_valid[1] - x_valid[0]),
                1e-4,
            )
        if derivative_right <= 0:
            derivative_right = max(
                (y_valid[-1] - y_valid[-2]) / (x_valid[-1] - x_valid[-2]),
                1e-4,
            )

        left_rate = derivative_left / value_left if value_left > 0 else 0.0
        right_rate = (
            derivative_right / (1 - value_right) if value_right < 1 else 0.0
        )

        def make_function(
            spline: PchipInterpolator,
            x_min: float,
            x_max: float,
            y_min: float,
            y_max: float,
            left_k: float,
            right_k: float,
        ) -> WinExpectancyFunction:
            def evaluate(score_difference: float | np.ndarray) -> np.ndarray:
                score_difference = np.asarray(score_difference, dtype=float)
                result = np.zeros_like(score_difference)
                left = score_difference < x_min
                right = score_difference > x_max
                middle = ~(left | right)
                if y_min <= 0:
                    result[left] = 0
                else:
                    result[left] = y_min * np.exp(
                        left_k * (score_difference[left] - x_min)
                    )
                result[middle] = spline(score_difference[middle])
                if y_max >= 1:
                    result[right] = 1
                else:
                    result[right] = 1 - (1 - y_max) * np.exp(
                        right_k * (x_max - score_difference[right])
                    )
                return np.clip(result, 0, 1)

            return evaluate

        condition = tuple(row["condition"])
        base_state = tuple(row["base_runners"])
        splines.setdefault(condition, {})[base_state] = make_function(
            raw_spline,
            x_left,
            x_right,
            value_left,
            value_right,
            left_rate,
            right_rate,
        )

    return splines


def get_win_expectancy(
    splines: dict[GameCondition, dict[BaseState, WinExpectancyFunction]],
    condition: GameCondition,
    base_state: BaseState,
    score_difference: float,
    *,
    terminal_result: int | None = None,
) -> float:
    """Return home-team Win Expectancy for a game state."""
    if terminal_result in (0, 1):
        return float(terminal_result)
    if condition[0] > 9:
        return 0.52
    if condition not in splines:
        raise ValueError(f"Unknown game condition: {condition}")
    if base_state not in splines[condition]:
        raise ValueError(f"Unknown base state {base_state} for {condition}")
    return float(splines[condition][base_state](score_difference))


def sigmoid_adjustment(delta_we: float, steepness: float = 4.0) -> float:
    """Return the ARBI multiplier alpha on the interval (0, 2)."""
    return float(2 / (1 + np.exp(-steepness * delta_we)))


def context_adjustment(delta_we: float, we_end: float, alpha: float) -> float:
    """Return the bell-shaped CRBI multiplier beta used in the notebooks."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    center = (1 + delta_we) / 2
    spread = min((center - delta_we) / 2, (1 - center) / 2)
    if spread <= 0:
        raise ValueError("delta_we must produce a positive context interval")
    return float(
        (2 / alpha) * np.exp(-((we_end - center) ** 2) / (2 * spread**2))
    )


def adjusted_rbi(rbi: float, delta_we: float, steepness: float = 4.0) -> float:
    """Calculate ARBI from RBI and the change in Win Expectancy."""
    return sigmoid_adjustment(delta_we, steepness) * rbi


def contextual_rbi(
    rbi: float,
    delta_we: float,
    we_end: float,
    steepness: float = 4.0,
) -> float:
    """Calculate CRBI from RBI, Win Expectancy change, and terminal WE."""
    alpha = sigmoid_adjustment(delta_we, steepness)
    beta = context_adjustment(delta_we, we_end, alpha)
    return beta * alpha * rbi


def add_adjusted_rbi_columns(
    events: pd.DataFrame,
    *,
    rbi_column: str = "RBI",
    delta_we_column: str = "delta_WE",
    we_end_column: str = "WE_end",
    steepness: float = 4.0,
) -> pd.DataFrame:
    """Return an event table with alpha, beta, ARBI, and CRBI columns."""
    required = {rbi_column, delta_we_column, we_end_column}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = events.copy()
    result["alpha"] = result[delta_we_column].map(
        lambda value: sigmoid_adjustment(value, steepness)
    )
    result["ARBI"] = result["alpha"] * result[rbi_column]
    result["beta"] = result.apply(
        lambda row: context_adjustment(
            row[delta_we_column], row[we_end_column], row["alpha"]
        ),
        axis=1,
    )
    result["CRBI"] = result["beta"] * result["ARBI"]
    return result
