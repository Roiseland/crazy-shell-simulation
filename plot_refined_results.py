"""Plots for the geometry-coupled 200 x 100 x 50 mm study."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


BLUE = "#3776AB"
LIGHT_BLUE = "#A9C7E1"
ORANGE = "#C9934B"
RED = "#B65745"
GRID = "#D8DEE7"
TEXT = "#202833"


def _font() -> None:
    for candidate in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.55, color=GRID)
    ax.set_axisbelow(True)


def create_refined_figure(study: dict, config: dict, output_stem: Path) -> None:
    _font()
    plt.rcParams.update({"font.size": 10.5, "text.color": TEXT, "axes.labelcolor": TEXT})
    geometry = study["geometry"]
    frequency = study["frequency"]
    waveform = study["waveform"]
    sensitivity = study["sensitivity"]
    backpressure = study["backpressure"]
    monte_carlo = study["monte_carlo"]

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.3))
    fig.subplots_adjust(left=0.075, right=0.96, bottom=0.13, top=0.83, wspace=0.30, hspace=0.43)
    fig.suptitle("200 × 100 × 50 mm 装置的几何耦合仿真", fontsize=23, y=0.965)
    fig.text(
        0.5,
        0.91,
        f"壁厚 3 mm；装配间隙 2 mm；V3 砂水泵统一缩放 {geometry.scale:.4f} 倍；固定步长 RK4",
        ha="center",
        fontsize=11.5,
    )

    # A: footprint and envelope.
    ax = axes[0, 0]
    ax.set_title("A  装置与缩放泵体包络（俯视）", loc="left")
    outer = FancyBboxPatch((0, 0), 200, 100, boxstyle="round,pad=0,rounding_size=8", ec=TEXT, fc="#F3F5F8", lw=1.8)
    internal = Rectangle((3, 3), 194, 94, ec="#768493", fc="none", lw=1.2, ls="--")
    pump_x = (200 - geometry.pump_length_mm) / 2
    pump_y = (100 - geometry.pump_width_mm) / 2
    pump = FancyBboxPatch(
        (pump_x, pump_y),
        geometry.pump_length_mm,
        geometry.pump_width_mm,
        boxstyle="round,pad=0,rounding_size=4",
        ec=BLUE,
        fc=LIGHT_BLUE,
        lw=2.0,
        alpha=0.85,
    )
    ax.add_patch(outer); ax.add_patch(internal); ax.add_patch(pump)
    ax.text(100, 106, "整机外廓 200 × 100 mm", ha="center", fontsize=10.5)
    ax.text(100, 50, f"泵体 {geometry.pump_length_mm:.1f} × {geometry.pump_width_mm:.1f} mm\n高度 {geometry.pump_height_mm:.1f} mm", ha="center", va="center", fontsize=11)
    ax.text(100, -9, "虚线：扣除 3 mm 壁厚后的内部边界", ha="center", color="#596779")
    ax.set_xlim(-8, 208); ax.set_ylim(-14, 114); ax.set_aspect("equal"); ax.axis("off")

    # B: converged waveform.
    ax = axes[0, 1]
    ax.plot(waveform["time_s"] * 1000, waveform["outlet_flow_ml_min"], color=BLUE, lw=2, label="出口瞬时流量")
    ax.fill_between(
        waveform["time_s"] * 1000,
        0,
        waveform["outlet_flow_ml_min"],
        where=waveform["outlet_flow_ml_min"] < 0,
        color=RED,
        alpha=0.16,
        label="回流区间",
    )
    mean_flow = frequency.loc[np.isclose(frequency.frequency_hz, 2.0), "net_flow_ml_min"].iloc[0]
    ax.axhline(mean_flow, color=TEXT, ls="--", lw=1.2, label=f"周期净流量 {mean_flow:.1f} mL/min")
    ax.set_title("B  2.0 Hz 稳态周期流量", loc="left")
    ax.set_xlabel("周期内时间（ms）"); ax.set_ylabel("流量（mL/min）")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax)

    # C: frequency sweep with uncertainty.
    ax = axes[1, 0]
    grouped = monte_carlo.groupby("frequency_hz")["sample_amount_30s_ml"]
    q05 = grouped.quantile(0.05); q50 = grouped.quantile(0.50); q95 = grouped.quantile(0.95)
    x = q50.index.to_numpy()
    ax.fill_between(x, q05.to_numpy(), q95.to_numpy(), color=LIGHT_BLUE, alpha=0.45, label="蒙特卡洛 90% 区间")
    ax.plot(frequency["frequency_hz"], frequency["sample_amount_30s_ml"], "-o", color=BLUE, lw=2.2, label="基准黏度 5 mPa·s")
    for _, row in frequency.iterrows():
        ax.text(row.frequency_hz, row.sample_amount_30s_ml + 4, f"{row.sample_amount_30s_ml:.1f}", ha="center", fontsize=9.5)
    for viscosity, group in sensitivity.groupby("viscosity_mpa_s"):
        if np.isclose(viscosity, 5.0):
            continue
        ax.plot(group["frequency_hz"], group["sample_amount_30s_ml"], "--", lw=1.2, alpha=0.75, label=f"{viscosity:g} mPa·s")
    ax.set_title("C  频率、黏度与 30 秒采样量", loc="left")
    ax.set_xlabel("驱动频率（Hz）"); ax.set_ylabel("净采样体积（mL）")
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    _style(ax)

    # D: pressure-flow response.
    ax = axes[1, 1]
    ax.plot(backpressure["backpressure_pa"], backpressure["net_flow_ml_min"], "-o", color=ORANGE, lw=2.2)
    ax.axhline(0, color=TEXT, lw=1)
    ax.fill_between(backpressure["backpressure_pa"], 0, backpressure["net_flow_ml_min"], color=ORANGE, alpha=0.18)
    flow = backpressure["net_flow_ml_min"].to_numpy()
    pressure = backpressure["backpressure_pa"].to_numpy()
    crossing = np.flatnonzero(np.signbit(flow[:-1]) != np.signbit(flow[1:]))
    if crossing.size:
        index = int(crossing[0])
        stall = pressure[index] + (0.0 - flow[index]) * (
            pressure[index + 1] - pressure[index]
        ) / (flow[index + 1] - flow[index])
        ax.axvline(stall, color=RED, ls="--", lw=1.3)
        ax.text(
            stall + 0.012,
            0.78 * float(np.max(flow)),
            f"截止背压约 {stall:.3f} Pa",
            color=RED,
            va="center",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
        )
    ax.set_title("D  2.0 Hz 背压—净流量关系", loc="left")
    ax.set_xlabel("出口背压（Pa）"); ax.set_ylabel("周期净流量（mL/min）")
    _style(ax)

    fig.text(
        0.075,
        0.045,
        "说明：矩形流道阻力、液柱惯性和方向性局部损失已进入时间域方程。膜片加载行程与损失系数沿用假设，结果用于设计筛查，需由 CFD 或样机压差—流量测试标定。",
        fontsize=10,
        color="#4B596B",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
