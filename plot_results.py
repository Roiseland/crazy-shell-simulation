"""Create the four-panel simulation summary figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


DARK_BLUE = "#3F73A8"
LIGHT_BLUE = "#A8C3DD"
SAND = "#C7A875"
GRID = "#D9DEE7"
TEXT = "#202631"


def _set_chinese_font() -> None:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False


def _labels(ax, bars, fmt="{:.1f}", offset=0.35) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=TEXT,
        )


def _style_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#475467")


def create_figure(summary: pd.DataFrame, config: dict, output_stem: Path) -> None:
    _set_chinese_font()
    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT})

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.1))
    fig.subplots_adjust(left=0.075, right=0.94, bottom=0.13, top=0.84, wspace=0.28, hspace=0.43)
    fig.suptitle(config["experiment"]["title"], fontsize=24, y=0.965)
    fig.text(
        0.5,
        0.915,
        f"最大开合角 {config['experiment']['max_shell_angle_deg']:.0f}°；驱动频率 1.0–2.5 Hz；每组模拟 {config['experiment']['repeats']} 次；随机种子 {config['experiment']['seed']}",
        ha="center",
        fontsize=11.5,
    )

    # Panel A
    ax = axes[0, 0]
    a = summary[(summary.panel == "A") & (summary.metric == "success_rate_pct")]
    x = np.arange(len(a))
    bars = ax.bar(x, a["mean"], width=0.62, color=[DARK_BLUE, LIGHT_BLUE, DARK_BLUE, LIGHT_BLUE])
    ax.set_title("A  不同底质条件下的采样成功率", loc="left", fontsize=11)
    ax.set_ylabel("采样成功率（%）")
    ax.set_xticks(x, a["condition_label"])
    ax.set_ylim(0, 105)
    _labels(ax, bars, fmt="{:.0f}%", offset=1.2)
    _style_axis(ax)

    # Panel B
    ax = axes[0, 1]
    b = summary[(summary.panel == "B") & (summary.metric == "sample_amount_ml")]
    freq = b["condition_id"].str.replace("_hz", "").astype(float).to_numpy()
    amount = b["mean"].to_numpy()
    error = b["sd"].to_numpy()
    ax.plot(freq, amount, color=DARK_BLUE, marker="o", linewidth=2.2, markersize=6)
    ax.errorbar(freq, amount, yerr=error, fmt="none", color="#61758A", capsize=4, linewidth=1.4)
    ax.fill_between(freq, amount - error, amount + error, color=LIGHT_BLUE, alpha=0.28)
    for xx, yy in zip(freq, amount):
        ax.text(xx, yy + 1.6, f"{yy:.1f}", ha="center", fontsize=9.5)
    ax.set_title("B  驱动频率与单次采样量", loc="left", fontsize=11)
    ax.set_xlabel("驱动频率（Hz）")
    ax.set_ylabel("单次采样量（mL）")
    ax.set_xlim(0.85, 2.65)
    ax.set_ylim(10, 40)
    _style_axis(ax)

    # Panel C
    ax = axes[1, 0]
    c_actual = summary[(summary.panel == "C") & (summary.metric == "achieved_depth_cm")]
    c_target = summary[(summary.panel == "C") & (summary.metric == "target_depth_cm")]
    x = np.arange(len(c_actual))
    width = 0.34
    bars1 = ax.bar(x - width / 2, c_actual["mean"], width, color=DARK_BLUE, label="仿真达到深度")
    bars2 = ax.bar(x + width / 2, c_target["mean"], width, color=SAND, label="设计目标深度")
    ax.errorbar(x - width / 2, c_actual["mean"], yerr=c_actual["sd"], fmt="none", color="#53697E", capsize=3)
    ax.set_title("C  典型场景的有效采样深度", loc="left", fontsize=11)
    ax.set_ylabel("有效采样深度（cm）")
    ax.set_xticks(x, c_actual["condition_label"])
    ax.set_ylim(0, 30)
    ax.legend(frameon=False, loc="upper left")
    _labels(ax, bars1, offset=0.35)
    _labels(ax, bars2, offset=0.35)
    _style_axis(ax)

    # Panel D uses separate axes because CV and energy have different units.
    ax = axes[1, 1]
    d_cv = summary[(summary.panel == "D") & (summary.metric == "repeatability_cv_pct")]
    d_energy = summary[(summary.panel == "D") & (summary.metric == "energy_wh")]
    x = np.arange(len(d_cv))
    width = 0.34
    bars1 = ax.bar(x - width / 2, d_cv["mean"], width, color=LIGHT_BLUE, label="采样量变异系数")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width / 2, d_energy["mean"], width, color=DARK_BLUE, label="单次能耗")
    ax.set_title("D  重复性与单次能耗", loc="left", fontsize=11)
    ax.set_ylabel("采样量变异系数（%）", color="#6688AA")
    ax2.set_ylabel("单次能耗（Wh）", color=DARK_BLUE)
    ax.set_xticks(x, d_cv["condition_label"])
    ax.set_ylim(0, 10.5)
    ax2.set_ylim(0, 10.5)
    handles = [bars1, bars2]
    ax.legend(handles, ["采样量变异系数", "单次能耗"], frameon=False, loc="upper left")
    _labels(ax, bars1, offset=0.18)
    _labels(ax2, bars2, offset=0.18)
    _style_axis(ax)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(axis="y", colors=DARK_BLUE)

    fig.text(
        0.075,
        0.045,
        "说明：结果由低阶参数模型生成，用于实验设计与程序验证。误差棒为模拟标准差；未经过样机数据标定，不代表实测性能。",
        fontsize=10.5,
        color="#475467",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
