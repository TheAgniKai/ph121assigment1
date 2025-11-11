"""Generate publication-ready figures for the PH121 assignment."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path
from statistics import mean
from typing import Any, List, Sequence, Tuple

from .examples import harmonic_oscillator_exact
from .report import AssignmentReport


class FigureGenerationError(RuntimeError):
    """Raised when matplotlib is unavailable for figure generation."""


@dataclass(frozen=True)
class SavedFigure:
    """Information about a saved matplotlib figure."""

    identifier: str
    path: Path


def _norm(vector: Sequence[float]) -> float:
    return sqrt(sum(component * component for component in vector))


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise FigureGenerationError("matplotlib is required to create figures") from exc
    return plt


def _save_figure(plt, figure, path: Path, *, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    if show:  # pragma: no cover - interactive display
        plt.show()
    plt.close(figure)


def _figure_harmonic_oscillator(report: AssignmentReport, plt) -> Tuple[str, Any]:
    oscillator = report.oscillator
    times = list(oscillator.result.times)
    positions = [state[0] for state in oscillator.result.states]
    velocities = [state[1] for state in oscillator.result.states]
    exact_positions = [harmonic_oscillator_exact(time)[0] for time in times]
    exact_velocities = [harmonic_oscillator_exact(time)[1] for time in times]
    position_errors = [abs(num - exact) for num, exact in zip(positions, exact_positions)]
    velocity_errors = [abs(num - exact) for num, exact in zip(velocities, exact_velocities)]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(times, positions, label="Numerical", linewidth=2.0)
    axes[0].plot(times, exact_positions, label="Analytic", linestyle="--")
    axes[0].set_ylabel("Position x(t)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(times, velocities, label="Numerical", linewidth=2.0)
    axes[1].plot(times, exact_velocities, label="Analytic", linestyle="--")
    axes[1].set_ylabel("Velocity v(t)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    axes[2].plot(times, position_errors, label="|Δx|")
    axes[2].plot(times, velocity_errors, label="|Δv|")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Absolute error")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    fig.suptitle("6.1 Harmonic oscillator accuracy check")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return "assignment-6.1-harmonic-oscillator", fig


def _figure_mercury_orbits(report: AssignmentReport, plt) -> Tuple[str, Any]:
    mercury = report.mercury
    rel_positions = mercury.relativistic.cartesian_positions()
    newt_positions = mercury.newtonian.cartesian_positions()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([x for x, _ in newt_positions], [y for _, y in newt_positions], label="Newtonian")
    ax.plot([x for x, _ in rel_positions], [y for _, y in rel_positions], label="Relativistic")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("6.2 Mercury orbit trajectories")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return "assignment-6.2-mercury-orbits", fig


def _figure_mercury_conservation(result, plt, label: str) -> Tuple[str, Any]:
    from .mercury import SECONDS_PER_YEAR

    energy0 = result.energies[0]
    angular0 = result.angular_momenta[0]
    energy_drift = [(energy - energy0) / abs(energy0) for energy in result.energies]
    angular_drift = [
        (angular - angular0) / abs(angular0) for angular in result.angular_momenta
    ]
    times = [time / SECONDS_PER_YEAR for time in result.times]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    ax1.plot(times, energy_drift)
    ax1.set_ylabel(r"$\Delta H / H_0$")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(times, angular_drift)
    ax2.set_ylabel(r"$\Delta L / L_0$")
    ax2.set_xlabel("Time [years]")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"6.2 Mercury conservation diagnostics – {label}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    identifier = f"assignment-6.2-mercury-{label.lower()}-conservation"
    identifier = identifier.replace(" ", "-")
    return identifier, fig


def _figure_lyapunov_growth(report: AssignmentReport, plt) -> Tuple[str, Any]:
    if not report.chaos.lyapunov_samples:
        raise FigureGenerationError("Lyapunov samples are required for plotting")
    sample = report.chaos.lyapunov_samples[0]
    times = list(sample.base_result.times)
    displacements: List[float] = []
    for base_state, perturbed_state in zip(sample.base_result.states, sample.perturbed_result.states):
        diff = [p - b for p, b in zip(perturbed_state, base_state)]
        displacements.append(_norm(diff))
    if not displacements or displacements[0] == 0.0:
        raise FigureGenerationError("Invalid displacement data for Lyapunov plot")
    growth = [log(value / displacements[0]) for value in displacements if value > 0.0]
    growth_times = [time for time, value in zip(times, displacements) if value > 0.0]
    slope = sample.exponent
    fit_line = [slope * (time - growth_times[0]) for time in growth_times]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(growth_times, growth, label=r"$\ln |\delta y(t)|/|\delta y(0)|$")
    ax.plot(growth_times, fit_line, label=rf"Slope $\lambda \approx {slope:.3f}$", linestyle="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Log displacement growth")
    ax.set_title("6.3.1 Lyapunov growth window")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return "assignment-6.3.1-lyapunov-growth", fig


def _figure_lyapunov_distribution(report: AssignmentReport, plt) -> Tuple[str, Any]:
    values = [estimate.exponent for estimate in report.chaos.lyapunov_samples]
    if not values:
        raise FigureGenerationError("No Lyapunov samples available")
    mean_value = report.chaos.lyapunov_mean

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = max(5, min(20, len(values)))
    ax.hist(values, bins=bins, color="#4477aa", alpha=0.75, edgecolor="black")
    ax.axvline(mean_value, color="#cc3311", linestyle="--", label=f"Mean = {mean_value:.3f}")
    ax.set_xlabel(r"$\lambda_{\max}$ [1/s]")
    ax.set_ylabel("Samples")
    ax.set_title("6.3.1 Distribution of maximal Lyapunov estimates")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return "assignment-6.3.1-lyapunov-distribution", fig


def _figure_lorenz_geometry(report: AssignmentReport, plt) -> Tuple[str, Any]:
    if not report.chaos.lyapunov_samples:
        raise FigureGenerationError("Lyapunov samples are required for Lorenz trajectory")
    trajectory = report.chaos.lyapunov_samples[0].base_result.states
    xs = [state[0] for state in trajectory]
    ys = [state[1] for state in trajectory]
    zs = [state[2] for state in trajectory]

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(xs, zs, linewidth=0.8)
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    ax1.set_title("Lorenz attractor (x, z)")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")  # type: ignore[attr-defined]
    ax2.plot(xs, ys, zs, color="#1f77b4", linewidth=0.7)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("Lorenz attractor trajectory")

    fig.suptitle("6.3 Chaotic Lorenz attractor")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return "assignment-6.3-lorenz-attractor", fig


def _linear_fit(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        raise ValueError("at least two points required")
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = 0.0
    denominator = 0.0
    for x, y in zip(xs, ys):
        numerator += (x - x_mean) * (y - y_mean)
        denominator += (x - x_mean) ** 2
    if denominator == 0.0:
        raise ValueError("zero variance in fit data")
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _figure_dimension_estimates(report: AssignmentReport, plt) -> Tuple[str, Any]:
    box_eps = [estimate.epsilon for estimate in report.chaos.box_counting]
    box_counts = [estimate.box_count for estimate in report.chaos.box_counting]
    info_eps = [estimate.epsilon for estimate in report.chaos.information]
    info_entropy = [estimate.entropy for estimate in report.chaos.information]

    box_xs = [log(1.0 / epsilon) for epsilon in box_eps if epsilon > 0.0]
    box_ys = [log(count) for epsilon, count in zip(box_eps, box_counts) if epsilon > 0.0 and count > 0]
    slope_box, intercept_box = _linear_fit(box_xs, box_ys)

    info_xs = [abs(log(epsilon)) for epsilon in info_eps if epsilon > 0.0]
    info_ys = [entropy for epsilon, entropy in zip(info_eps, info_entropy) if epsilon > 0.0]
    slope_info, intercept_info = _linear_fit(info_xs, info_ys)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(box_xs, box_ys, color="#4477aa", label="Samples")
    fit_box = [slope_box * x + intercept_box for x in sorted(box_xs)]
    ax1.plot(sorted(box_xs), fit_box, color="#cc3311", linestyle="--", label=f"Fit slope ≈ {slope_box:.3f}")
    ax1.set_xlabel(r"$\ln (1/\varepsilon)$")
    ax1.set_ylabel(r"$\ln N(\varepsilon)$")
    ax1.set_title("6.3.2 Box-counting dimension")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.scatter(info_xs, info_ys, color="#228833", label="Entropy samples")
    fit_info = [slope_info * x + intercept_info for x in sorted(info_xs)]
    ax2.plot(sorted(info_xs), fit_info, color="#cc3311", linestyle="--", label=f"Fit slope ≈ {slope_info:.3f}")
    ax2.set_xlabel(r"$|\ln \varepsilon|$")
    ax2.set_ylabel("Entropy")
    ax2.set_title("6.3.2 Information dimension")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    fig.tight_layout()
    return "assignment-6.3.2-dimensions", fig


def _figure_lyapunov_spectrum(report: AssignmentReport, plt) -> Tuple[str, Any]:
    exponents = list(report.chaos.spectrum.exponents)
    if not exponents:
        raise FigureGenerationError("Lyapunov spectrum is empty")
    kaplan_yorke = report.chaos.spectrum.kaplan_yorke_dimension
    indices = list(range(len(exponents)))
    labels = [rf"$\lambda_{idx}$" for idx in indices]
    cumulative: List[float] = []
    total = 0.0
    for value in exponents:
        total += value
        cumulative.append(total)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(indices, exponents, color="#4477aa")
    ax1.axhline(0.0, color="black", linewidth=1.0)
    ax1.set_xticks(indices)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel(r"$\lambda$ [1/s]")
    ax1.set_title("6.3.3 Lyapunov spectrum")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(indices, cumulative, marker="o")
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.axvline(kaplan_yorke, color="#cc3311", linestyle="--", label=rf"$D_{{KY}} \approx {kaplan_yorke:.2f}$")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Cumulative sum")
    ax2.set_title("Cumulative Lyapunov sums")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    fig.tight_layout()
    return "assignment-6.3.3-lyapunov-spectrum", fig


def _figure_cross_sections(report: AssignmentReport, plt) -> Tuple[str, Any]:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

    ee = report.electron_muon
    ax1.errorbar([0], [ee.result.value], yerr=[ee.result.error], fmt="o", color="#4477aa", label="Monte Carlo")
    if ee.analytic is not None:
        ax1.axhline(ee.analytic, color="#cc3311", linestyle="--", label="Analytic")
    ax1.set_ylabel(r"$\sigma$ [pb]")
    ax1.set_title(r"6 Monte Carlo: $e^+e^- \to \mu^+\mu^-$")
    ax1.set_xticks([])
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    dy8 = report.drell_yan_8tev
    dy14 = report.drell_yan_14tev
    energies = [8, 14]
    values = [dy8.result.value, dy14.result.value]
    errors = [dy8.result.error, dy14.result.error]
    reference = [dy8.analytic or 0.0, dy14.analytic or 0.0]
    ax2.bar(energies, values, yerr=errors, color="#228833", alpha=0.8, width=1.0, label="Monte Carlo")
    ax2.plot(energies, reference, color="#cc3311", linestyle="--", marker="o", label="Benchmark")
    ax2.set_xlabel(r"$\sqrt{s}$ [TeV]")
    ax2.set_ylabel(r"$\sigma$ [pb]")
    ax2.set_title("6 Drell–Yan total cross sections")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    histogram = report.drell_yan_histogram
    bin_edges = list(histogram.bin_edges)
    bin_values = list(histogram.bin_values)
    bin_errors = list(histogram.bin_errors)
    if len(bin_edges) >= 2:
        centres: List[float] = []
        widths: List[float] = []
        for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
            centres.append(0.5 * (lower + upper))
            widths.append(upper - lower)
        ax3.bar(centres, bin_values, width=widths, color="#4477aa", edgecolor="black", align="center")
        ax3.errorbar(centres, bin_values, yerr=bin_errors, fmt="none", ecolor="#cc3311", capsize=3)
        ax3.set_xlabel(r"$\tau$")
        ax3.set_ylabel(r"$d\sigma/d\tau$ [pb]")
        ax3.set_title("6 Drell–Yan differential cross section (8 TeV)")
        ax3.set_yscale("log")
        ax3.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    return "assignment-6-monte-carlo", fig


FIGURE_BUILDERS = (
    _figure_harmonic_oscillator,
    _figure_mercury_orbits,
    lambda report, plt: _figure_mercury_conservation(report.mercury.relativistic, plt, "Relativistic"),
    lambda report, plt: _figure_mercury_conservation(report.mercury.newtonian, plt, "Newtonian"),
    _figure_lyapunov_growth,
    _figure_lyapunov_distribution,
    _figure_lorenz_geometry,
    _figure_dimension_estimates,
    _figure_lyapunov_spectrum,
    _figure_cross_sections,
)


def create_assignment_figures(
    report: AssignmentReport,
    directory: str | Path,
    *,
    show: bool = False,
) -> Tuple[SavedFigure, ...]:
    """Create and save all assignment figures.

    Parameters
    ----------
    report:
        The :class:`~ph121.report.AssignmentReport` containing simulation data.
    directory:
        Target directory for the generated figures. Created if necessary.
    show:
        When ``True`` the figures are displayed using :func:`matplotlib.pyplot.show`
        after saving. The default ``False`` keeps the function non-interactive.
    """

    plt = _ensure_matplotlib()
    output_dir = Path(directory)
    saved: List[SavedFigure] = []
    for builder in FIGURE_BUILDERS:
        identifier, figure = builder(report, plt)  # type: ignore[arg-type]
        path = output_dir / f"{identifier}.png"
        _save_figure(plt, figure, path, show=show)
        saved.append(SavedFigure(identifier=identifier, path=path))
    return tuple(saved)


__all__ = [
    "FigureGenerationError",
    "SavedFigure",
    "create_assignment_figures",
]

