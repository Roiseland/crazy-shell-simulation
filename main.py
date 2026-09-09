"""Run the simulation and create data and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_results import create_figure
from simulation import load_config, run_simulation, summarise_results


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行疯狂的贝壳采样性能仿真")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)

    trials = run_simulation(config)
    summary = summarise_results(trials, config)
    trials.to_csv(args.output / "trial_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output / "summary.csv", index=False, encoding="utf-8-sig")
    create_figure(summary, config, args.output / "simulation_results")

    print(f"完成：{len(trials)} 条模拟记录")
    print(f"汇总：{args.output / 'summary.csv'}")
    print(f"图表：{args.output / 'simulation_results.png'}")


if __name__ == "__main__":
    main()
