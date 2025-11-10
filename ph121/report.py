"""Produce a consolidated report for the numerical integration assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .chaos import (
    BoxCountingEstimate,
    InformationDimensionEstimate,
    LyapunovEstimate,
    LyapunovSpectrumEstimate,
    estimate_lorenz_box_counting_dimension,
    estimate_lorenz_information_dimension,
    estimate_lorenz_lyapunov_spectrum,
    estimate_lorenz_maximal_lyapunov,
)
from .examples import OscillatorAnalysis, integrate_harmonic_oscillator
from .high_energy import (
    DifferentialHistogram,
    drell_yan_differential_cross_section,
    drell_yan_total_cross_section,
    total_cross_section_electron_muon,
)
from .mercury import PrecessionEstimate, estimate_precession
from .monte_carlo import MonteCarloResult


def _logspace(start: float, stop: float, count: int, *, base: float = 10.0) -> List[float]:
    """Return ``count`` logarithmically spaced values including the end points."""

    if count < 2:
        raise ValueError("count must be at least 2")
    if base <= 0.0:
        raise ValueError("base must be positive")
    step = (stop - start) / float(count - 1)
    exponent = start
    values: List[float] = []
    for _ in range(count):
        values.append(base ** exponent)
        exponent += step
    return values


@dataclass(frozen=True)
class ChaosReport:
    """Container for the chaotic attractor measurements."""

    lyapunov_samples: Tuple[LyapunovEstimate, ...]
    lyapunov_mean: float
    lyapunov_stdev: float
    box_counting: Tuple[BoxCountingEstimate, ...]
    box_dimension: float
    information: Tuple[InformationDimensionEstimate, ...]
    information_dimension: float
    spectrum: LyapunovSpectrumEstimate


@dataclass(frozen=True)
class MonteCarloSummary:
    """Capture Monte Carlo estimates along with optional analytic values."""

    result: MonteCarloResult
    analytic: float | None = None


@dataclass(frozen=True)
class AssignmentReport:
    """Aggregate the outputs for each assignment task."""

    oscillator: OscillatorAnalysis
    mercury: PrecessionEstimate
    chaos: ChaosReport
    electron_muon: MonteCarloSummary
    drell_yan_8tev: MonteCarloSummary
    drell_yan_14tev: MonteCarloSummary
    drell_yan_histogram_total: MonteCarloSummary
    drell_yan_histogram: DifferentialHistogram

def _format_value(value: float, error: float | None = None) -> str:
    if error is None:
        return f"{value:.6g}"
    return f"{value:.6g} +/- {error:.2g}"


def generate_report(
    *,
    oscillator_periods: float = 10.0,
    oscillator_step: float = 0.01,
    precession_years: float = 100.0,
    precession_step_days: float = 0.25,
    lyapunov_samples: int = 8,
    lyapunov_step: float = 0.01,
    lyapunov_warmup: float = 20.0,
    lyapunov_eval: float = 10.0,
    chaos_epsilons: Sequence[float] | None = None,
    information_epsilons: Sequence[float] | None = None,
    mc_samples: int = 20000,
    drell_yan_bins: Iterable[float] | None = None,
) -> AssignmentReport:
    """Run every numerical experiment with sensible defaults."""

    oscillator = integrate_harmonic_oscillator(periods=oscillator_periods, step=oscillator_step)
    mercury = estimate_precession(duration_years=precession_years, step_days=precession_step_days)

    estimates, mean_value, stdev_value = estimate_lorenz_maximal_lyapunov(
        samples=lyapunov_samples,
        warmup_time=lyapunov_warmup,
        evaluation_time=lyapunov_eval,
        step=lyapunov_step,
    )

    if chaos_epsilons is None:
        chaos_epsilons = _logspace(1.5, -0.5, 8)
    box_counts, box_dimension = estimate_lorenz_box_counting_dimension(chaos_epsilons)

    if information_epsilons is None:
        information_epsilons = _logspace(-0.2, -1.8, 8)
    information, info_dimension = estimate_lorenz_information_dimension(information_epsilons)

    spectrum = estimate_lorenz_lyapunov_spectrum(
        warmup_time=lyapunov_warmup,
        sample_time=3.0 * lyapunov_eval,
        step=lyapunov_step,
    )

    electron_muon_result, analytic = total_cross_section_electron_muon(energy=200.0, samples=mc_samples)

    dy8_result = drell_yan_total_cross_section(energy=8000.0, samples=mc_samples)
    dy14_result = drell_yan_total_cross_section(energy=14000.0, samples=mc_samples)

    if drell_yan_bins is None:
        tau_min = (60.0 * 60.0) / (8000.0 * 8000.0)
        edges: List[float] = [tau_min]
        edges.extend(_logspace(-3.5, -1.0, 5))
        edges.extend([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
        drell_yan_bins = sorted(set(edges))
    hist_total, histogram = drell_yan_differential_cross_section(
        energy=8000.0,
        samples=mc_samples,
        bins=drell_yan_bins,
    )

    chaos_report = ChaosReport(
        lyapunov_samples=tuple(estimates),
        lyapunov_mean=mean_value,
        lyapunov_stdev=stdev_value,
        box_counting=tuple(box_counts),
        box_dimension=box_dimension,
        information=tuple(information),
        information_dimension=info_dimension,
        spectrum=spectrum,
    )

    return AssignmentReport(
        oscillator=oscillator,
        mercury=mercury,
        chaos=chaos_report,
        electron_muon=MonteCarloSummary(result=electron_muon_result, analytic=analytic),
        drell_yan_8tev=MonteCarloSummary(result=dy8_result, analytic=881.8),
        drell_yan_14tev=MonteCarloSummary(result=dy14_result, analytic=1684.0),
        drell_yan_histogram_total=MonteCarloSummary(result=hist_total, analytic=881.8),
        drell_yan_histogram=histogram,
    )


def format_report(report: AssignmentReport) -> str:
    """Render a human-readable version of the :class:`AssignmentReport`."""

    lines: List[str] = []

    oscillator = report.oscillator
    lines.append("=== Check-in: Harmonic oscillator ===")
    final_time = oscillator.result.times[-1]
    final_state = oscillator.result.states[-1]
    step = oscillator.result.times[1] - oscillator.result.times[0] if len(oscillator.result.times) > 1 else 0.0
    lines.append(f"Integrated for t = {final_time:.3f} s with dt = {step:.3f} s")
    lines.append(f"Final state (x, v) = ({final_state[0]:.6f}, {final_state[1]:.6f})")
    lines.append(
        "Maximum errors dx = "
        + _format_value(oscillator.max_position_error)
        + ", dv = "
        + _format_value(oscillator.max_velocity_error)
    )
    lines.append("")

    mercury = report.mercury
    lines.append("=== Relativistic precession of Mercury ===")
    lines.append(
        "Estimated advance = "
        + _format_value(mercury.arcseconds_per_century)
        + " arcsec/century"
    )
    lines.append(
        f"Integrated {len(mercury.relativistic.times)} samples with step {mercury.relativistic.step / 86400.0:.3f} days"
    )
    lines.append("")

    chaos = report.chaos
    lines.append("=== Chaotic attractor diagnostics (Lorenz 63) ===")
    lines.append(
        f"Maximal Lyapunov exponent lambda_max = {chaos.lyapunov_mean:.4f} +/- {chaos.lyapunov_stdev:.4f} 1/s"
        f" from {len(chaos.lyapunov_samples)} samples"
    )
    lines.append(f"Box-counting dimension D0 ≈ {chaos.box_dimension:.4f}")
    lines.append(f"Information dimension D1 ≈ {chaos.information_dimension:.4f}")
    spectrum_values = ", ".join(f"{value:.4f}" for value in chaos.spectrum.exponents)
    lines.append(f"Lyapunov spectrum = [{spectrum_values}]")
    lines.append(
        f"Kaplan-Yorke dimension DL ≈ {chaos.spectrum.kaplan_yorke_dimension:.4f}"
    )
    lines.append("")

    lines.append("=== Monte Carlo cross sections ===")
    ee_result = report.electron_muon
    lines.append(
        "sigma(e+ e- -> mu+ mu-) = "
        + _format_value(ee_result.result.value, ee_result.result.error)
        + " (analytic "
        + _format_value(ee_result.analytic or 0.0)
        + ")"
    )
    dy8 = report.drell_yan_8tev
    lines.append(
        "sigma(pp -> gamma/Z -> mu+ mu-; sqrt(s) = 8 TeV) = "
        + _format_value(dy8.result.value, dy8.result.error)
        + " pb (benchmark 881.8 pb)"
    )
    dy14 = report.drell_yan_14tev
    lines.append(
        "sigma(pp -> gamma/Z -> mu+ mu-; sqrt(s) = 14 TeV) = "
        + _format_value(dy14.result.value, dy14.result.error)
        + " pb (benchmark 1684 pb)"
    )
    hist_total = report.drell_yan_histogram_total
    lines.append(
        "Histogram integral (sqrt(s) = 8 TeV) = "
        + _format_value(hist_total.result.value, hist_total.result.error)
        + " pb"
    )
    lines.append("d sigma/d tau histogram (sqrt(s) = 8 TeV):")
    histogram = report.drell_yan_histogram
    for lower, upper, value, error in zip(
        histogram.bin_edges[:-1],
        histogram.bin_edges[1:],
        histogram.bin_values,
        histogram.bin_errors,
    ):
        lines.append(
            f"  tau in [{lower:.3g}, {upper:.3g}]: {_format_value(value, error)} pb"
        )

    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    report = generate_report()
    print(format_report(report))


__all__ = [
    "AssignmentReport",
    "ChaosReport",
    "MonteCarloSummary",
    "format_report",
    "generate_report",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    main()
