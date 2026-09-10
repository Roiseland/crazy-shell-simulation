"""Geometry-coupled time-domain model for the 200 x 100 x 50 mm device.

The V3 B slurry-pump geometry is uniformly scaled into the available internal
envelope. A prescribed chamber-volume waveform drives a lumped hydraulic
network with rectangular-duct resistance, fluid inertance and asymmetric
quadratic losses. This is a planning model; its loss coefficients and loaded
stroke still require experimental or CFD calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScaledGeometry:
    scale: float
    outer_length_mm: float
    outer_width_mm: float
    outer_height_mm: float
    internal_length_mm: float
    internal_width_mm: float
    internal_height_mm: float
    usable_length_mm: float
    usable_width_mm: float
    usable_height_mm: float
    pump_length_mm: float
    pump_width_mm: float
    pump_height_mm: float
    fluid_height_mm: float
    port_length_mm: float
    port_width_mm: float
    main_channel_width_mm: float
    rectifier_length_mm: float
    side_gap_mm: float
    scaled_swept_volume_ml_cycle: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HydraulicNetwork:
    resistance_in_pa_s_m3: float
    resistance_out_pa_s_m3: float
    inertance_in_pa_s2_m3: float
    inertance_out_pa_s2_m3: float
    minimum_flow_area_m2: float
    port_flow_area_m2: float
    hydraulic_diameter_m: float


def build_scaled_geometry(config: dict) -> ScaledGeometry:
    device = config["device"]
    source = config["source_geometry"]
    outer = np.asarray(device["outer_dimensions_mm"], dtype=float)
    wall = float(device["wall_thickness_mm"])
    clearance = float(device["assembly_clearance_mm"])
    internal = outer - 2.0 * wall
    usable = internal - 2.0 * clearance
    source_bbox = np.asarray(source["assembly_dimensions_mm"], dtype=float)
    if np.any(usable <= 0):
        raise ValueError("Wall thickness and clearance leave no usable internal space.")
    scale = float(np.min(usable / source_bbox))
    pump = source_bbox * scale
    swept = config["actuation"]["reference_swept_volume_ml_cycle"] * scale**3
    return ScaledGeometry(
        scale=scale,
        outer_length_mm=outer[0],
        outer_width_mm=outer[1],
        outer_height_mm=outer[2],
        internal_length_mm=internal[0],
        internal_width_mm=internal[1],
        internal_height_mm=internal[2],
        usable_length_mm=usable[0],
        usable_width_mm=usable[1],
        usable_height_mm=usable[2],
        pump_length_mm=pump[0],
        pump_width_mm=pump[1],
        pump_height_mm=pump[2],
        fluid_height_mm=(source["cavity_depth_mm"] + source["gasket_height_mm"]) * scale,
        port_length_mm=source["port_length_mm"] * scale,
        port_width_mm=source["port_width_mm"] * scale,
        main_channel_width_mm=source["main_channel_width_mm"] * scale,
        rectifier_length_mm=source["rectifier_length_mm"] * scale,
        side_gap_mm=source["side_gap_mm"] * scale,
        scaled_swept_volume_ml_cycle=swept,
    )


def _rectangular_duct(
    length_m: float, width_m: float, height_m: float, viscosity: float, density: float
) -> tuple[float, float]:
    width_m, height_m = max(width_m, height_m), min(width_m, height_m)
    aspect_correction = max(1.0 - 0.63 * height_m / width_m, 0.1)
    resistance = 12.0 * viscosity * length_m / (
        width_m * height_m**3 * aspect_correction
    )
    inertance = density * length_m / (width_m * height_m)
    return resistance, inertance


def build_network(
    geometry: ScaledGeometry, config: dict, viscosity_pa_s: float
) -> HydraulicNetwork:
    fluid = config["fluid"]
    density = float(fluid["density_kg_m3"])
    scale = geometry.scale
    source = config["source_geometry"]
    height = geometry.fluid_height_mm * 1e-3
    port_length = geometry.port_length_mm * 1e-3
    port_width = geometry.port_width_mm * 1e-3
    full_width = geometry.main_channel_width_mm * 1e-3
    gap_width = geometry.side_gap_mm * 1e-3
    half_rectifier = 0.5 * geometry.rectifier_length_mm * 1e-3

    pre_inlet_mm = max(
        0.0,
        20.0 - source["port_length_mm"],
    ) * scale
    post_outlet_mm = max(
        0.0,
        source["body_length_mm"]
        - source["port_length_mm"]
        - 20.0
        - source["rectifier_length_mm"],
    ) * scale

    r_port, m_port = _rectangular_duct(
        port_length, port_width, height, viscosity_pa_s, density
    )
    r_pre, m_pre = _rectangular_duct(
        pre_inlet_mm * 1e-3, full_width, height, viscosity_pa_s, density
    )
    r_post, m_post = _rectangular_duct(
        post_outlet_mm * 1e-3, full_width, height, viscosity_pa_s, density
    )
    r_gap_single, m_gap_single = _rectangular_duct(
        half_rectifier, gap_width, height, viscosity_pa_s, density
    )
    # The two side passages around the rectifier operate in parallel.
    r_gap = r_gap_single / 2.0
    m_gap = m_gap_single / 2.0
    area_gap = 2.0 * gap_width * height
    area_port = port_width * height
    wetted_width = 2.0 * (2.0 * gap_width + height)
    hydraulic_diameter = 4.0 * area_gap / wetted_width
    return HydraulicNetwork(
        resistance_in_pa_s_m3=r_port + r_pre + r_gap,
        resistance_out_pa_s_m3=r_port + r_post + r_gap,
        inertance_in_pa_s2_m3=m_port + m_pre + m_gap,
        inertance_out_pa_s2_m3=m_port + m_post + m_gap,
        minimum_flow_area_m2=min(area_gap, area_port),
        port_flow_area_m2=area_port,
        hydraulic_diameter_m=hydraulic_diameter,
    )


def simulate_condition(
    config: dict,
    geometry: ScaledGeometry,
    frequency_hz: float,
    viscosity_pa_s: float,
    backpressure_pa: float = 0.0,
    swept_volume_factor: float = 1.0,
    loss_factor: float = 1.0,
    points_per_cycle: int | None = None,
    settling_cycles: int | None = None,
    waveform: bool = False,
) -> dict:
    fluid = config["fluid"]
    actuation = config["actuation"]
    network = build_network(geometry, config, viscosity_pa_s)
    density = float(fluid["density_kg_m3"])
    port_k = float(fluid["port_loss_coefficient"])
    k_forward = float(fluid["forward_loss_coefficient"]) * loss_factor
    k_reverse = float(fluid["reverse_loss_coefficient"]) * loss_factor
    points = int(points_per_cycle or actuation["points_per_cycle"])
    cycles = int(settling_cycles or actuation["settling_cycles"])
    omega = 2.0 * math.pi * frequency_hz
    period = 1.0 / frequency_hz
    dt = period / points
    swept_m3 = geometry.scaled_swept_volume_ml_cycle * 1e-6 * swept_volume_factor
    volume_amplitude = swept_m3 / 2.0
    area = network.minimum_flow_area_m2
    m_in = network.inertance_in_pa_s2_m3
    m_out = network.inertance_out_pa_s2_m3
    r_in = network.resistance_in_pa_s_m3
    r_out = network.resistance_out_pa_s_m3

    def volume_rate(t: float) -> tuple[float, float]:
        return (
            volume_amplitude * omega * math.cos(omega * t),
            -volume_amplitude * omega**2 * math.sin(omega * t),
        )

    def loss(q: float) -> float:
        directional_k = k_forward if q >= 0.0 else k_reverse
        return 0.5 * density * (port_k + directional_k) * q * abs(q) / area**2

    def derivative(t: float, q_out: float) -> float:
        source, source_derivative = volume_rate(t)
        q_in = q_out + source
        return (
            -m_in * source_derivative
            - r_in * q_in
            - r_out * q_out
            - loss(q_in)
            - loss(q_out)
            - backpressure_pa
        ) / (m_in + m_out)

    q_out = 0.0
    final_t: list[float] = []
    final_q: list[float] = []
    total_steps = cycles * points
    for step in range(total_steps):
        t = step * dt
        if step >= total_steps - points:
            final_t.append(t)
            final_q.append(q_out)
        k1 = derivative(t, q_out)
        k2 = derivative(t + dt / 2.0, q_out + dt * k1 / 2.0)
        k3 = derivative(t + dt / 2.0, q_out + dt * k2 / 2.0)
        k4 = derivative(t + dt, q_out + dt * k3)
        q_out += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    final_t.append(total_steps * dt)
    final_q.append(q_out)

    time = np.asarray(final_t)
    time -= time[0]
    q_out_array = np.asarray(final_q)
    phase_time = time
    source = volume_amplitude * omega * np.cos(omega * phase_time)
    source_derivative = -volume_amplitude * omega**2 * np.sin(omega * phase_time)
    q_in_array = q_out_array + source
    loss_in = np.array([loss(value) for value in q_in_array])
    loss_out = np.array([loss(value) for value in q_out_array])
    dq_out = np.array([derivative(t, q) for t, q in zip(phase_time, q_out_array)])
    pressure_chamber = -(m_in * (dq_out + source_derivative) + r_in * q_in_array + loss_in)
    pressure_residual = pressure_chamber - (
        backpressure_pa + m_out * dq_out + r_out * q_out_array + loss_out
    )
    mean_flow_m3_s = float(np.trapezoid(q_out_array, phase_time) / period)
    sample_amount_ml = mean_flow_m3_s * actuation["sample_duration_s"] * 1e6
    peak_velocity = max(np.max(np.abs(q_in_array)), np.max(np.abs(q_out_array))) / area
    reynolds = density * peak_velocity * network.hydraulic_diameter_m / viscosity_pa_s
    hydraulic_power_w = abs(float(np.trapezoid(-pressure_chamber * source, phase_time) / period))

    result = {
        "frequency_hz": frequency_hz,
        "viscosity_mpa_s": viscosity_pa_s * 1000.0,
        "backpressure_pa": backpressure_pa,
        "swept_volume_ml_cycle": swept_m3 * 1e6,
        "net_flow_ml_min": mean_flow_m3_s * 60.0e6,
        "sample_amount_30s_ml": sample_amount_ml,
        "outlet_min_ml_min": float(np.min(q_out_array) * 60.0e6),
        "outlet_max_ml_min": float(np.max(q_out_array) * 60.0e6),
        "chamber_pressure_half_range_pa": float(
            (np.max(pressure_chamber) - np.min(pressure_chamber)) / 2.0
        ),
        "peak_reynolds_number": float(reynolds),
        "hydraulic_input_power_mw": hydraulic_power_w * 1000.0,
        "pressure_residual_pa": float(np.max(np.abs(pressure_residual))),
        "points_per_cycle": points,
        "settling_cycles": cycles,
    }
    if waveform:
        result["waveform"] = pd.DataFrame(
            {
                "time_s": phase_time,
                "inlet_flow_ml_min": q_in_array * 60.0e6,
                "outlet_flow_ml_min": q_out_array * 60.0e6,
                "chamber_pressure_pa": pressure_chamber,
                "volume_rate_ml_min": source * 60.0e6,
            }
        )
    return result


def run_refined_study(config: dict) -> dict:
    geometry = build_scaled_geometry(config)
    actuation = config["actuation"]
    fluid = config["fluid"]
    baseline_viscosity = float(fluid["baseline_viscosity_pa_s"])

    frequency_rows = []
    baseline_waveform = None
    for frequency in map(float, actuation["frequencies_hz"]):
        result = simulate_condition(
            config,
            geometry,
            frequency,
            baseline_viscosity,
            waveform=frequency == 2.0,
        )
        if "waveform" in result:
            baseline_waveform = result.pop("waveform")
        frequency_rows.append(result)

    sensitivity_rows = []
    for viscosity in map(float, fluid["viscosity_scan_pa_s"]):
        for frequency in map(float, actuation["frequencies_hz"]):
            result = simulate_condition(config, geometry, frequency, viscosity)
            sensitivity_rows.append(result)

    backpressure_rows = []
    for pressure in map(float, config["backpressure_scan_pa"]):
        backpressure_rows.append(
            simulate_condition(config, geometry, 2.0, baseline_viscosity, pressure)
        )

    uncertainty = config["uncertainty"]
    rng = np.random.default_rng(int(uncertainty["seed"]))
    mc_rows = []
    for frequency in map(float, actuation["frequencies_hz"]):
        for trial in range(int(uncertainty["repeats"])):
            viscosity = baseline_viscosity * math.exp(
                rng.normal(0.0, uncertainty["viscosity_relative_sd"])
            )
            swept_factor = max(
                0.5, rng.normal(1.0, uncertainty["swept_volume_relative_sd"])
            )
            loss_factor = max(
                0.5, rng.normal(1.0, uncertainty["loss_coefficient_relative_sd"])
            )
            result = simulate_condition(
                config,
                geometry,
                frequency,
                viscosity,
                swept_volume_factor=swept_factor,
                loss_factor=loss_factor,
                points_per_cycle=int(uncertainty["monte_carlo_points_per_cycle"]),
                settling_cycles=int(uncertainty["monte_carlo_settling_cycles"]),
            )
            result["trial"] = trial + 1
            result["swept_volume_factor"] = swept_factor
            result["loss_factor"] = loss_factor
            mc_rows.append(result)

    convergence_rows = []
    for points in (100, 200, 400, 800):
        result = simulate_condition(
            config,
            geometry,
            2.0,
            baseline_viscosity,
            points_per_cycle=points,
            settling_cycles=int(actuation["settling_cycles"]),
        )
        result["convergence_type"] = "points_per_cycle"
        convergence_rows.append(result)
    for cycles in (15, 30, 45, 60, 90):
        result = simulate_condition(
            config,
            geometry,
            2.0,
            baseline_viscosity,
            points_per_cycle=int(actuation["points_per_cycle"]),
            settling_cycles=cycles,
        )
        result["convergence_type"] = "settling_cycles"
        convergence_rows.append(result)

    return {
        "geometry": geometry,
        "frequency": pd.DataFrame(frequency_rows),
        "waveform": baseline_waveform,
        "sensitivity": pd.DataFrame(sensitivity_rows),
        "backpressure": pd.DataFrame(backpressure_rows),
        "monte_carlo": pd.DataFrame(mc_rows),
        "convergence": pd.DataFrame(convergence_rows),
    }
