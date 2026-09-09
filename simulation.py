"""Parameterised, low-order sampler simulation.

The model is intended for experiment planning. Parameters in config.json are
assumptions until they are calibrated against bench or field measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PumpModel:
    swept_volume_ml_per_cycle: float
    rectification_efficiency: float
    fill_time_constant_s: float
    resonance_frequency_hz: float
    resonance_gain: float
    resonance_width_hz: float

    def nominal_amount_ml(self, frequency_hz: float, duration_s: float) -> float:
        """Return retained fluid before applying a substrate factor."""
        cycles = frequency_hz * duration_s
        ideal_displacement = self.swept_volume_ml_per_cycle * cycles
        fill_factor = 1.0 / (1.0 + self.fill_time_constant_s * frequency_hz)
        offset = (frequency_hz - self.resonance_frequency_hz) / self.resonance_width_hz
        response_gain = 1.0 + self.resonance_gain * np.exp(-0.5 * offset**2)
        return ideal_displacement * self.rectification_efficiency * fill_factor * response_gain


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _amount_cv(frequency_hz: float) -> float:
    """Assumed relative variability, lowest near the 2 Hz operating point."""
    below_optimum = max(2.0 - frequency_hz, 0.0)
    above_optimum = max(frequency_hz - 2.0, 0.0)
    return 0.048 + 0.014 * below_optimum**2 + 0.004 * above_optimum


def _sample_positive_normal(
    rng: np.random.Generator, mean: float, relative_sd: float, size: int
) -> np.ndarray:
    values = rng.normal(mean, abs(mean) * relative_sd, size=size)
    return np.maximum(values, 0.0)


def _stratified_uniform(rng: np.random.Generator, size: int) -> np.ndarray:
    """Return one random point per equal-probability interval.

    Stratification keeps a 100-run reliability experiment from being dominated
    by binomial sampling noise while retaining random trial ordering.
    """
    values = (np.arange(size) + rng.random(size)) / size
    rng.shuffle(values)
    return values


def run_simulation(config: dict) -> pd.DataFrame:
    exp = config["experiment"]
    pump = PumpModel(**config["pump"])
    power = config["power"]
    repeats = int(exp["repeats"])
    rng = np.random.default_rng(int(exp["seed"]))
    rows: list[dict] = []

    # Substrate experiment at the nominal 2 Hz operating point.
    nominal_frequency = float(pump.resonance_frequency_hz)
    base_amount = pump.nominal_amount_ml(nominal_frequency, exp["sample_duration_s"])
    for substrate in config["substrates"]:
        amount = _sample_positive_normal(
            rng,
            base_amount * substrate["retention_factor"],
            _amount_cv(nominal_frequency),
            repeats,
        )
        depth = _sample_positive_normal(
            rng,
            15.0 * substrate["penetration_ratio"],
            0.04,
            repeats,
        )
        completed = _stratified_uniform(rng, repeats) < substrate["operation_success_probability"]
        successful = (
            completed
            & (amount >= exp["success_min_amount_ml"])
            & (depth >= exp["success_min_depth_cm"])
        )
        for trial in range(repeats):
            rows.append(
                {
                    "experiment": "substrate",
                    "condition_id": substrate["id"],
                    "condition_label": substrate["label"],
                    "trial": trial + 1,
                    "frequency_hz": nominal_frequency,
                    "sample_amount_ml": amount[trial],
                    "effective_depth_cm": depth[trial],
                    "success": bool(successful[trial]),
                    "energy_wh": np.nan,
                    "target_depth_cm": 15.0,
                }
            )

    # Frequency experiment uses a single reference substrate.
    for frequency in map(float, exp["frequencies_hz"]):
        mean_amount = pump.nominal_amount_ml(frequency, exp["sample_duration_s"])
        amounts = _sample_positive_normal(rng, mean_amount, _amount_cv(frequency), repeats)
        mean_power_w = (
            power["idle_power_w"]
            + power["linear_power_w_per_hz"] * frequency
            + power["quadratic_power_w_per_hz2"] * frequency**2
        )
        mean_energy = mean_power_w * exp["full_cycle_duration_s"] / 3600.0
        energy = _sample_positive_normal(
            rng, mean_energy, power["relative_noise"], repeats
        )
        for trial in range(repeats):
            rows.append(
                {
                    "experiment": "frequency",
                    "condition_id": f"{frequency:.1f}_hz",
                    "condition_label": f"{frequency:.1f} Hz",
                    "trial": trial + 1,
                    "frequency_hz": frequency,
                    "sample_amount_ml": amounts[trial],
                    "effective_depth_cm": np.nan,
                    "success": np.nan,
                    "energy_wh": energy[trial],
                    "target_depth_cm": np.nan,
                }
            )

    # Scenario experiment compares achieved depth against its design target.
    for scenario in config["scenarios"]:
        target = float(scenario["target_depth_cm"])
        mean_depth = target * scenario["achievement_ratio"]
        depths = _sample_positive_normal(
            rng, mean_depth, scenario["relative_noise"], repeats
        )
        depths = np.minimum(depths, target)
        for trial in range(repeats):
            rows.append(
                {
                    "experiment": "scenario",
                    "condition_id": scenario["id"],
                    "condition_label": scenario["label"],
                    "trial": trial + 1,
                    "frequency_hz": nominal_frequency,
                    "sample_amount_ml": np.nan,
                    "effective_depth_cm": depths[trial],
                    "success": np.nan,
                    "energy_wh": np.nan,
                    "target_depth_cm": target,
                }
            )

    result = pd.DataFrame(rows)
    _validate_results(result, config)
    return result


def _validate_results(results: pd.DataFrame, config: dict) -> None:
    expected = int(config["experiment"]["repeats"])
    counts = results.groupby(["experiment", "condition_id"]).size()
    if not (counts == expected).all():
        raise ValueError("Every condition must contain the configured number of trials.")
    numeric = results[["sample_amount_ml", "effective_depth_cm", "energy_wh"]]
    if (numeric.fillna(0) < 0).any().any():
        raise ValueError("Physical output variables cannot be negative.")
    scenario = results[results["experiment"] == "scenario"]
    if (scenario["effective_depth_cm"] > scenario["target_depth_cm"] + 1e-12).any():
        raise ValueError("Scenario depth exceeded its configured mechanical target.")


def summarise_results(results: pd.DataFrame, config: dict) -> pd.DataFrame:
    summaries: list[dict] = []

    substrate = results[results["experiment"] == "substrate"]
    for condition_id, group in substrate.groupby("condition_id", sort=False):
        summaries.append(
            {
                "panel": "A",
                "condition_id": condition_id,
                "condition_label": group["condition_label"].iloc[0],
                "metric": "success_rate_pct",
                "mean": 100.0 * group["success"].astype(bool).mean(),
                "sd": np.nan,
            }
        )

    frequency = results[results["experiment"] == "frequency"]
    for condition_id, group in frequency.groupby("condition_id", sort=False):
        amount_mean = group["sample_amount_ml"].mean()
        amount_sd = group["sample_amount_ml"].std(ddof=1)
        summaries.extend(
            [
                {
                    "panel": "B",
                    "condition_id": condition_id,
                    "condition_label": group["condition_label"].iloc[0],
                    "metric": "sample_amount_ml",
                    "mean": amount_mean,
                    "sd": amount_sd,
                },
                {
                    "panel": "D",
                    "condition_id": condition_id,
                    "condition_label": group["condition_label"].iloc[0],
                    "metric": "repeatability_cv_pct",
                    "mean": 100.0 * amount_sd / amount_mean,
                    "sd": np.nan,
                },
                {
                    "panel": "D",
                    "condition_id": condition_id,
                    "condition_label": group["condition_label"].iloc[0],
                    "metric": "energy_wh",
                    "mean": group["energy_wh"].mean(),
                    "sd": group["energy_wh"].std(ddof=1),
                },
            ]
        )

    scenario = results[results["experiment"] == "scenario"]
    for condition_id, group in scenario.groupby("condition_id", sort=False):
        summaries.extend(
            [
                {
                    "panel": "C",
                    "condition_id": condition_id,
                    "condition_label": group["condition_label"].iloc[0],
                    "metric": "achieved_depth_cm",
                    "mean": group["effective_depth_cm"].mean(),
                    "sd": group["effective_depth_cm"].std(ddof=1),
                },
                {
                    "panel": "C",
                    "condition_id": condition_id,
                    "condition_label": group["condition_label"].iloc[0],
                    "metric": "target_depth_cm",
                    "mean": group["target_depth_cm"].iloc[0],
                    "sd": np.nan,
                },
            ]
        )

    return pd.DataFrame(summaries)
