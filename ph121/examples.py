"""Example problems solved with the :mod:`ph121.integrators` helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, tau
from typing import Tuple

from .integrators import IntegrationResult, runge_kutta4


@dataclass(frozen=True)
class OscillatorAnalysis:
    """Summary of the harmonic oscillator integration accuracy."""

    result: IntegrationResult
    max_position_error: float
    max_velocity_error: float


def harmonic_oscillator_derivative(_: float, state: Tuple[float, float]) -> Tuple[float, float]:
    """Time derivative for a unit-frequency harmonic oscillator."""

    position, velocity = state
    return velocity, -position


def harmonic_oscillator_exact(t: float) -> Tuple[float, float]:
    """Analytic solution with initial conditions x(0)=1, v(0)=0."""

    return cos(t), -sin(t)


def integrate_harmonic_oscillator(*, periods: float = 1.0, step: float = 0.01) -> OscillatorAnalysis:
    """Integrate a harmonic oscillator and report the maximum error.

    Parameters
    ----------
    periods:
        Number of oscillation periods :math:`2\pi` to evolve for.
    step:
        Integration time step.
    """

    t_end = periods * tau
    result = runge_kutta4(
        harmonic_oscillator_derivative,
        (1.0, 0.0),
        t0=0.0,
        t_end=t_end,
        step=step,
    )

    max_pos_err = 0.0
    max_vel_err = 0.0
    for t, (position, velocity) in zip(result.times, result.states):
        exact_position, exact_velocity = harmonic_oscillator_exact(t)
        max_pos_err = max(max_pos_err, abs(position - exact_position))
        max_vel_err = max(max_vel_err, abs(velocity - exact_velocity))

    return OscillatorAnalysis(
        result=result,
        max_position_error=max_pos_err,
        max_velocity_error=max_vel_err,
    )


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def main(periods: float = 1.0, step: float = 0.01) -> None:
    """Integrate and print the harmonic oscillator accuracy summary."""

    analysis = integrate_harmonic_oscillator(periods=periods, step=step)
    final_time = analysis.result.times[-1]
    final_state = analysis.result.states[-1]
    print(
        "Integrated harmonic oscillator for",
        _format_float(final_time),
        "seconds using dt =",
        _format_float(step),
    )
    print("Final state (x, v) =", tuple(_format_float(v) for v in final_state))
    print(
        "Maximum errors: Δx =",
        _format_float(analysis.max_position_error),
        ", Δv =",
        _format_float(analysis.max_velocity_error),
    )


if __name__ == "__main__":  # pragma: no cover - simple demonstration helper
    main()
