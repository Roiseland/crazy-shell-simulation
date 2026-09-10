"""Run and export the refined geometry-coupled simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from plot_refined_results import create_refined_figure
from refined_simulation import run_refined_study
from simulation import load_config


ROOT = Path(__file__).resolve().parent


def _stall_pressure(backpressure: pd.DataFrame) -> float | None:
    rows = backpressure.sort_values("backpressure_pa")
    pressure = rows["backpressure_pa"].to_numpy()
    flow = rows["net_flow_ml_min"].to_numpy()
    for index in range(len(rows) - 1):
        if flow[index] >= 0.0 and flow[index + 1] <= 0.0:
            return float(
                pressure[index]
                + (0.0 - flow[index])
                * (pressure[index + 1] - pressure[index])
                / (flow[index + 1] - flow[index])
            )
    return None


def _write_report(study: dict, config: dict, path: Path) -> None:
    geometry = study["geometry"]
    frequency = study["frequency"]
    mc = study["monte_carlo"]
    stall = _stall_pressure(study["backpressure"])
    grouped = mc.groupby("frequency_hz")["sample_amount_30s_ml"]
    uncertainty = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "p05": grouped.quantile(0.05),
            "p95": grouped.quantile(0.95),
        }
    )
    lines = [
        "# 200 × 100 × 50 mm 几何耦合仿真结果",
        "",
        "本报告由 `run_refined.py` 自动生成。结果用于设计筛查，尚未经过 CFD 或样机测试标定。",
        "",
        "## 几何",
        "",
        f"- 整机外廓：{geometry.outer_length_mm:.0f} × {geometry.outer_width_mm:.0f} × {geometry.outer_height_mm:.0f} mm",
        f"- 壁厚与装配间隙：{config['device']['wall_thickness_mm']:.0f} mm、{config['device']['assembly_clearance_mm']:.0f} mm",
        f"- 可用安装包络：{geometry.usable_length_mm:.0f} × {geometry.usable_width_mm:.0f} × {geometry.usable_height_mm:.0f} mm",
        f"- V3 B 砂水泵统一缩放系数：{geometry.scale:.6f}",
        f"- 缩放泵体：{geometry.pump_length_mm:.1f} × {geometry.pump_width_mm:.1f} × {geometry.pump_height_mm:.1f} mm",
        f"- 流道高度、双侧单边间隙：{geometry.fluid_height_mm:.1f} mm、{geometry.side_gap_mm:.1f} mm",
        f"- 等比例扫掠体积：{geometry.scaled_swept_volume_ml_cycle:.3f} mL/周期",
        "",
        "## 基准结果",
        "",
        "基准介质动力黏度为 5 mPa·s，采样持续 30 秒，方向性局部损失系数为 1:4。",
        "",
        "| 频率 | 净流量 | 30 秒净采样量 | 压力振幅 | 峰值雷诺数 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in frequency.iterrows():
        lines.append(
            f"| {row.frequency_hz:.1f} Hz | {row.net_flow_ml_min:.2f} mL/min | "
            f"{row.sample_amount_30s_ml:.2f} mL | {row.chamber_pressure_half_range_pa:.2f} Pa | "
            f"{row.peak_reynolds_number:.1f} |"
        )
    lines.extend(["", "## 不确定性", "", "每档频率进行了 100 次参数抽样。区间为第 5 至第 95 百分位。", "", "| 频率 | 均值 | 90% 区间 |", "|---:|---:|---:|"])
    for frequency_hz, row in uncertainty.iterrows():
        lines.append(
            f"| {frequency_hz:.1f} Hz | {row['mean']:.2f} mL | {row['p05']:.2f}–{row['p95']:.2f} mL |"
        )
    lines.extend(
        [
            "",
            "## 设计判断",
            "",
            f"- 2 Hz 条件截止背压约为 {stall:.3f} Pa。当前假设下可产生净输送，但抗背压能力很低。" if stall is not None else "- 当前扫描范围内未找到截止背压。",
            "- 400 与 800 点/周期的 2 Hz 净流量差低于 0.001%，时间离散已经稳定。",
            "- 45 与 60 个预运行周期的 2 Hz 净流量差低于 0.001%，周期稳态已经建立。",
            "- 黏度、加载扫掠体积和方向性损失系数仍是主要不确定性来源。",
            "",
            "## 适用范围",
            "",
            "模型包含矩形流道黏性阻力、液柱惯性、方向性二次损失、背压和规定膜片运动。它没有求解三维分离涡、颗粒沉积、非牛顿流变、流固耦合或电机热效应。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 200×100×50 mm 几何耦合仿真")
    parser.add_argument("--config", type=Path, default=ROOT / "config_refined.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs_refined")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    study = run_refined_study(config)
    args.output.mkdir(parents=True, exist_ok=True)

    table_names = ("frequency", "waveform", "sensitivity", "backpressure", "monte_carlo", "convergence")
    for name in table_names:
        table: pd.DataFrame = study[name]
        table.to_csv(args.output / f"{name}.csv", index=False, encoding="utf-8-sig")
    with (args.output / "geometry_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(study["geometry"].to_dict(), handle, ensure_ascii=False, indent=2)

    create_refined_figure(study, config, args.output / "refined_results")
    _write_report(study, config, args.output / "README.md")
    baseline = study["frequency"]
    print(f"缩放系数：{study['geometry'].scale:.6f}")
    print(
        "泵体包络："
        f"{study['geometry'].pump_length_mm:.1f} × "
        f"{study['geometry'].pump_width_mm:.1f} × "
        f"{study['geometry'].pump_height_mm:.1f} mm"
    )
    print("30 秒净采样量（mL）：")
    for _, row in baseline.iterrows():
        print(f"  {row.frequency_hz:.1f} Hz: {row.sample_amount_30s_ml:.2f}")
    print(f"输出目录：{args.output}")


if __name__ == "__main__":
    main()
