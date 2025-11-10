"""Visualization helpers for the PH121 numerical assignment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence, Tuple

from .examples import OscillatorAnalysis, harmonic_oscillator_exact
from .mercury import (
    MERCURY_SEMIMAJOR_AXIS,
    PrecessionEstimate,
    SECONDS_PER_YEAR,
)

if TYPE_CHECKING:  # pragma: no cover - import-time helper
    from matplotlib.figure import Figure

    from .high_energy import DifferentialHistogram
    from .report import AssignmentReport, ChaosReport, MonteCarloSummary


def _require_matplotlib():
    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required to generate figures") from exc
    return plt


def _fractional_changes(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    baseline = values[0]
    if baseline == 0.0:
        return [0.0 for _ in values]
    return [(value - baseline) / abs(baseline) for value in values]


def plot_harmonic_oscillator_solution(
    analysis: OscillatorAnalysis,
    *,
    show: bool = True,
) -> "Figure":
    """Plot the harmonic oscillator trajectory, errors, and energy drift."""

    plt = _require_matplotlib()
    times = list(analysis.result.times)
    positions = [state[0] for state in analysis.result.states]
    velocities = [state[1] for state in analysis.result.states]
    exact_positions: List[float] = []
    exact_velocities: List[float] = []
    energies: List[float] = []
    for time_value, position, velocity in zip(times, positions, velocities):
        exact_position, exact_velocity = harmonic_oscillator_exact(time_value)
        exact_positions.append(exact_position)
        exact_velocities.append(exact_velocity)
        energies.append(0.5 * (position * position + velocity * velocity))
    baseline_energy = energies[0] if energies else 0.0
    energy_error = [energy - baseline_energy for energy in energies]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 10))
    ax_pos, ax_vel, ax_energy = axes
    ax_pos.plot(times, positions, label="Numerical")
    ax_pos.plot(times, exact_positions, "--", label="Analytic")
    ax_pos.set_ylabel("Position")
    ax_pos.legend()
    ax_pos.grid(True, linestyle="--", alpha=0.5)

    ax_vel.plot(times, velocities, label="Numerical")
    ax_vel.plot(times, exact_velocities, "--", label="Analytic")
    ax_vel.set_ylabel("Velocity")
    ax_vel.legend()
    ax_vel.grid(True, linestyle="--", alpha=0.5)

    ax_energy.plot(times, energy_error)
    ax_energy.set_ylabel(r"$\Delta E$")
    ax_energy.set_xlabel("Time [s]")
    ax_energy.grid(True, linestyle="--", alpha=0.5)
    ax_energy.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    fig.suptitle("Harmonic Oscillator Integration Summary")

    if show:  # pragma: no cover - plotting side effect
        plt.show()
    return fig


def plot_mercury_orbits(
    estimate: PrecessionEstimate,
    *,
    show: bool = True,
) -> "Figure":
    """Plot orbit geometry and conservation diagnostics for Mercury."""

    plt = _require_matplotlib()
    rel_positions = estimate.relativistic.cartesian_positions()
    newt_positions = estimate.newtonian.cartesian_positions()
    rel_radii = [radius / MERCURY_SEMIMAJOR_AXIS for radius in estimate.relativistic.radii]
    newt_radii = [radius / MERCURY_SEMIMAJOR_AXIS for radius in estimate.newtonian.radii]
    rel_times = [time / SECONDS_PER_YEAR for time in estimate.relativistic.times]
    newt_times = [time / SECONDS_PER_YEAR for time in estimate.newtonian.times]

    rel_energy = _fractional_changes(list(estimate.relativistic.energies))
    newt_energy = _fractional_changes(list(estimate.newtonian.energies))
    rel_angular = _fractional_changes(list(estimate.relativistic.angular_momenta))
    newt_angular = _fractional_changes(list(estimate.newtonian.angular_momenta))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_orbit, ax_radius, ax_energy, ax_angular = (
        axes[0][0],
        axes[0][1],
        axes[1][0],
        axes[1][1],
    )

    if rel_positions:
        rel_x = [x / MERCURY_SEMIMAJOR_AXIS for x, _ in rel_positions]
        rel_y = [y / MERCURY_SEMIMAJOR_AXIS for _, y in rel_positions]
        ax_orbit.plot(rel_x, rel_y, label="Relativistic")
    if newt_positions:
        newt_x = [x / MERCURY_SEMIMAJOR_AXIS for x, _ in newt_positions]
        newt_y = [y / MERCURY_SEMIMAJOR_AXIS for _, y in newt_positions]
        ax_orbit.plot(newt_x, newt_y, label="Newtonian", linestyle="--")
    ax_orbit.set_aspect("equal", adjustable="box")
    ax_orbit.set_xlabel("x [AU]")
    ax_orbit.set_ylabel("y [AU]")
    ax_orbit.legend()
    ax_orbit.set_title("Mercury Orbit Trajectories")
    ax_orbit.grid(True, linestyle="--", alpha=0.5)

    ax_radius.plot(rel_times, rel_radii, label="Relativistic")
    ax_radius.plot(newt_times, newt_radii, label="Newtonian", linestyle="--")
    ax_radius.set_xlabel("Time [years]")
    ax_radius.set_ylabel("Radius / a")
    ax_radius.legend()
    ax_radius.grid(True, linestyle="--", alpha=0.5)
    ax_radius.set_title("Radial Distance vs Time")

    ax_energy.plot(rel_times, rel_energy, label="Relativistic")
    ax_energy.plot(newt_times, newt_energy, label="Newtonian", linestyle="--")
    ax_energy.set_xlabel("Time [years]")
    ax_energy.set_ylabel(r"$\Delta H / H_0$")
    ax_energy.grid(True, linestyle="--", alpha=0.5)
    ax_energy.legend()

    ax_angular.plot(rel_times, rel_angular, label="Relativistic")
    ax_angular.plot(newt_times, newt_angular, label="Newtonian", linestyle="--")
    ax_angular.set_xlabel("Time [years]")
    ax_angular.set_ylabel(r"$\Delta L / L_0$")
    ax_angular.grid(True, linestyle="--", alpha=0.5)
    ax_angular.legend()

    fig.suptitle("Mercury Orbit Comparison")

    if show:  # pragma: no cover - plotting side effect
        plt.show()
    return fig


def plot_mercury_perihelion_progress(
    estimate: PrecessionEstimate,
    *,
    show: bool = True,
) -> "Figure":
    """Visualize perihelion angles and their relativistic excess."""

    plt = _require_matplotlib()
    rel_arcsec = [angle * 180.0 / math.pi * 3600.0 for angle in estimate.relativistic_perihelia]
    newt_arcsec = [angle * 180.0 / math.pi * 3600.0 for angle in estimate.newtonian_perihelia]
    rel_offsets = [value - rel_arcsec[0] for value in rel_arcsec] if rel_arcsec else []
    newt_offsets = [value - newt_arcsec[0] for value in newt_arcsec] if newt_arcsec else []
    delta = [r - n for r, n in zip(rel_offsets, newt_offsets)]
    indices = list(range(len(delta)))

    fig, (ax_absolute, ax_delta) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    orbit_numbers = list(range(len(rel_arcsec)))
    if rel_arcsec:
        ax_absolute.plot(orbit_numbers, rel_arcsec, label="Relativistic")
    if newt_arcsec:
        ax_absolute.plot(orbit_numbers, newt_arcsec, label="Newtonian", linestyle="--")
    ax_absolute.set_ylabel("Perihelion angle [arcsec]")
    ax_absolute.legend()
    ax_absolute.grid(True, linestyle="--", alpha=0.5)

    ax_delta.plot(indices, delta, marker="o")
    ax_delta.set_xlabel("Orbit index")
    ax_delta.set_ylabel("Relativistic excess [arcsec]")
    ax_delta.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Mercury Perihelion Advance")

    if show:  # pragma: no cover
        plt.show()
    return fig


def plot_chaos_diagnostics(
    report: "ChaosReport",
    *,
    sample_index: int = 0,
    show: bool = True,
) -> "Figure":
    """Plot Lorenz attractor projections and diagnostic fits."""

    if not report.lyapunov_samples:
        raise ValueError("At least one Lyapunov sample is required")
    if sample_index < 0 or sample_index >= len(report.lyapunov_samples):
        raise IndexError("sample_index out of range")

    plt = _require_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # type: ignore  # pragma: no cover

    sample = report.lyapunov_samples[sample_index]
    xs = [state[0] for state in sample.base_result.states]
    ys = [state[1] for state in sample.base_result.states]
    zs = [state[2] for state in sample.base_result.states]
    times = list(sample.base_result.times)

    displacements: List[float] = []
    for base_state, perturbed_state in zip(sample.base_result.states, sample.perturbed_result.states):
        displacements.append(math.dist(base_state, perturbed_state))
    initial = displacements[0] if displacements else 1.0
    log_growth = [math.log(value / initial) for value in displacements if value > 0.0]
    growth_times = times[: len(log_growth)]
    fit_line = [sample.exponent * time for time in growth_times]

    fig = plt.figure(figsize=(13, 12))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.0, 1.0])
    ax_attractor = fig.add_subplot(grid[0, 0:2], projection="3d")
    ax_timeseries = fig.add_subplot(grid[0, 2])
    ax_growth = fig.add_subplot(grid[1, 0:2])
    ax_hist = fig.add_subplot(grid[1, 2])
    ax_box = fig.add_subplot(grid[2, 0])
    ax_info = fig.add_subplot(grid[2, 1])
    ax_spectrum = fig.add_subplot(grid[2, 2])

    ax_attractor.plot(xs, ys, zs, linewidth=0.8)
    ax_attractor.set_title("Lorenz Attractor")
    ax_attractor.set_xlabel("x")
    ax_attractor.set_ylabel("y")
    ax_attractor.set_zlabel("z")

    ax_timeseries.plot(times, xs, label="x")
    ax_timeseries.plot(times, ys, label="y")
    ax_timeseries.plot(times, zs, label="z")
    ax_timeseries.set_xlabel("Time [s]")
    ax_timeseries.set_ylabel("State")
    ax_timeseries.legend()
    ax_timeseries.grid(True, linestyle="--", alpha=0.5)

    ax_growth.plot(growth_times, log_growth, label=r"$\ln |\delta y(t)|/|\delta y(0)|$")
    ax_growth.plot(growth_times, fit_line, linestyle="--", label=f"Fit slope = {sample.exponent:.3f}")
    ax_growth.set_xlabel("Time [s]")
    ax_growth.set_ylabel("Log displacement")
    ax_growth.legend()
    ax_growth.grid(True, linestyle="--", alpha=0.5)

    exponents = [estimate.exponent for estimate in report.lyapunov_samples]
    ax_hist.hist(exponents, bins=min(10, len(exponents)), color="#4472c4", alpha=0.7)
    ax_hist.axvline(report.lyapunov_mean, color="black", linestyle="--", label="Mean")
    ax_hist.set_xlabel(r"$\lambda_{\max}$ [1/s]")
    ax_hist.set_ylabel("Count")
    ax_hist.legend()

    box_eps = [math.log(1.0 / entry.epsilon) for entry in report.box_counting if entry.epsilon > 0.0]
    box_counts = [math.log(entry.box_count) for entry in report.box_counting if entry.box_count > 0]
    if box_eps and box_counts:
        ax_box.scatter(box_eps, box_counts)
        mean_x = sum(box_eps) / len(box_eps)
        mean_y = sum(box_counts) / len(box_counts)
        slope = report.box_dimension
        intercept = mean_y - slope * mean_x
        fit_x = [min(box_eps), max(box_eps)]
        fit_y = [slope * value + intercept for value in fit_x]
        ax_box.plot(fit_x, fit_y, linestyle="--", label=f"D0 = {slope:.3f}")
    ax_box.set_xlabel(r"$\log(1/\epsilon)$")
    ax_box.set_ylabel(r"$\log N(\epsilon)$")
    ax_box.grid(True, linestyle="--", alpha=0.5)
    ax_box.legend()

    info_eps = [abs(math.log(entry.epsilon)) for entry in report.information if entry.epsilon > 0.0]
    info_entropy = [entry.entropy for entry in report.information]
    if info_eps and info_entropy:
        ax_info.scatter(info_eps, info_entropy)
        mean_x = sum(info_eps) / len(info_eps)
        mean_y = sum(info_entropy) / len(info_entropy)
        slope = report.information_dimension
        intercept = mean_y - slope * mean_x
        fit_x = [min(info_eps), max(info_eps)]
        fit_y = [slope * value + intercept for value in fit_x]
        ax_info.plot(fit_x, fit_y, linestyle="--", label=f"D1 = {slope:.3f}")
    ax_info.set_xlabel(r"$|\log \epsilon|$")
    ax_info.set_ylabel("Entropy")
    ax_info.grid(True, linestyle="--", alpha=0.5)
    ax_info.legend()

    spectrum = report.spectrum.exponents
    ax_spectrum.bar(range(1, len(spectrum) + 1), spectrum, color="#70ad47")
    ax_spectrum.set_xlabel("Index")
    ax_spectrum.set_ylabel("Lyapunov exponent [1/s]")
    ax_spectrum.axhline(0.0, color="black", linewidth=0.8)
    ax_spectrum.set_title(f"Kaplan–Yorke D_L = {report.spectrum.kaplan_yorke_dimension:.3f}")

    fig.suptitle("Lorenz 63 Chaotic Diagnostics")

    if show:  # pragma: no cover
        plt.show()
    return fig


def plot_monte_carlo_progress(
    summary: "MonteCarloSummary",
    *,
    label: str,
    show: bool = True,
) -> "Figure":
    """Plot cumulative Monte Carlo convergence and contribution spread."""

    result = summary.result
    if result.weighted_samples is None:
        raise ValueError("MonteCarloResult does not include sample history")

    plt = _require_matplotlib()
    contributions = list(result.weighted_samples)
    sample_indices = list(range(1, len(contributions) + 1))
    cumulative: List[float] = []
    errors: List[float] = []
    total = 0.0
    total_sq = 0.0
    for index, value in enumerate(contributions, start=1):
        total += value
        total_sq += value * value
        mean = total / index
        cumulative.append(mean)
        if index > 1:
            variance = max((total_sq - index * mean * mean) / (index - 1), 0.0)
            errors.append(math.sqrt(variance / index))
        else:
            errors.append(0.0)

    fig, (ax_progress, ax_hist) = plt.subplots(2, 1, figsize=(9, 10))
    ax_progress.plot(sample_indices, cumulative, label="Estimate")
    lower = [value - err for value, err in zip(cumulative, errors)]
    upper = [value + err for value, err in zip(cumulative, errors)]
    ax_progress.fill_between(sample_indices, lower, upper, alpha=0.3, label="±1σ")
    if summary.analytic is not None:
        ax_progress.axhline(summary.analytic, color="black", linestyle="--", label="Benchmark")
    ax_progress.set_xlabel("Samples")
    ax_progress.set_ylabel("Cross section [pb]")
    ax_progress.set_title(f"Monte Carlo convergence: {label}")
    ax_progress.legend()
    ax_progress.grid(True, linestyle="--", alpha=0.5)

    ax_hist.hist(contributions, bins=40, color="#ed7d31", alpha=0.7)
    ax_hist.set_xlabel("Contribution per sample [pb]")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()

    if show:  # pragma: no cover
        plt.show()
    return fig


def plot_drell_yan_histogram(
    histogram: "DifferentialHistogram",
    *,
    show: bool = True,
) -> "Figure":
    """Plot the differential cross section histogram dσ/dτ."""

    plt = _require_matplotlib()
    bin_edges = list(histogram.bin_edges)
    bin_values = list(histogram.bin_values)
    bin_errors = list(histogram.bin_errors)
    centers: List[float] = []
    widths: List[float] = []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        centers.append(0.5 * (lower + upper))
        widths.append(upper - lower)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(centers, bin_values, width=widths, align="center", color="#5b9bd5", alpha=0.8, yerr=bin_errors, capsize=4)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$d\sigma/d\tau$ [pb]")
    ax.set_title("Drell–Yan differential cross section (√s = 8 TeV)")
    ax.set_xlim(min(bin_edges), max(bin_edges))
    ax.grid(True, linestyle="--", alpha=0.5)

    if show:  # pragma: no cover
        plt.show()
    return fig


def render_report_figures(
    report: "AssignmentReport",
    *,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> List[Path]:
    """Render figures for every assignment component and optionally save them."""

    plt = _require_matplotlib()
    figures: List[Tuple[str, "Figure"]] = []
    figures.append(("oscillator_summary", plot_harmonic_oscillator_solution(report.oscillator, show=False)))
    figures.append(("mercury_orbits", plot_mercury_orbits(report.mercury, show=False)))
    figures.append(("mercury_perihelion", plot_mercury_perihelion_progress(report.mercury, show=False)))
    figures.append(("lorenz_diagnostics", plot_chaos_diagnostics(report.chaos, show=False)))
    figures.append((
        "electron_muon_monte_carlo",
        plot_monte_carlo_progress(report.electron_muon, label=r"$e^+e^- \to \mu^+\mu^-$", show=False),
    ))
    figures.append((
        "drell_yan_8tev_monte_carlo",
        plot_monte_carlo_progress(report.drell_yan_8tev, label="pp → μ⁺μ⁻ at 8 TeV", show=False),
    ))
    figures.append((
        "drell_yan_14tev_monte_carlo",
        plot_monte_carlo_progress(report.drell_yan_14tev, label="pp → μ⁺μ⁻ at 14 TeV", show=False),
    ))
    figures.append(("drell_yan_histogram", plot_drell_yan_histogram(report.drell_yan_histogram, show=False)))

    saved_paths: List[Path] = []
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for name, figure in figures:
            file_path = output_path / f"{name}.png"
            figure.savefig(file_path, dpi=200, bbox_inches="tight")
            saved_paths.append(file_path)

    if show:  # pragma: no cover
        plt.show()
    else:
        for _, figure in figures:
            plt.close(figure)

    return saved_paths


__all__ = [
    "plot_harmonic_oscillator_solution",
    "plot_mercury_orbits",
    "plot_mercury_perihelion_progress",
    "plot_chaos_diagnostics",
    "plot_monte_carlo_progress",
    "plot_drell_yan_histogram",
    "render_report_figures",
]

