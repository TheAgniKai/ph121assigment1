"""Small harmonic oscillator demo for the integration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, tau
from typing import List, Tuple

from .integrators import IntegrationResult, runge_kutta4


@dataclass
class OscillatorAnalysis:
    """Hold the integration output and simple error checks."""

    result: IntegrationResult
    max_position_error: float
    max_velocity_error: float


def harmonic_oscillator_derivative(_: float, state: Tuple[float, float]) -> Tuple[float, float]:
    position, velocity = state
    return velocity, -position


def harmonic_oscillator_exact(t: float) -> Tuple[float, float]:
    return cos(t), -sin(t)


def integrate_harmonic_oscillator(*, periods: float = 1.0, step: float = 0.01) -> OscillatorAnalysis:
    total_time = periods * tau
    result = runge_kutta4(
        harmonic_oscillator_derivative,
        (1.0, 0.0),
        t0=0.0,
        t_end=total_time,
        step=step,
    )
    max_pos = 0.0
    max_vel = 0.0
    for idx, time in enumerate(result.times):
        state = result.states[idx]
        exact_state = harmonic_oscillator_exact(time)
        pos_error = abs(state[0] - exact_state[0])
        vel_error = abs(state[1] - exact_state[1])
        if pos_error > max_pos:
            max_pos = pos_error
        if vel_error > max_vel:
            max_vel = vel_error
    return OscillatorAnalysis(result=result, max_position_error=max_pos, max_velocity_error=max_vel)


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def main(periods: float = 1.0, step: float = 0.01) -> None:
    analysis = integrate_harmonic_oscillator(periods=periods, step=step)
    final_time = analysis.result.times[-1]
    final_state = analysis.result.states[-1]
    text: List[str] = []
    text.append("Integrated harmonic oscillator for")
    text.append(_format_float(final_time))
    text.append("seconds using dt =")
    text.append(_format_float(step))
    print(" ".join(text))
    pair = tuple(_format_float(value) for value in final_state)
    print("Final state (x, v) =", pair)
    print(
        "Maximum errors: Δx =",
        _format_float(analysis.max_position_error),
        ", Δv =",
        _format_float(analysis.max_velocity_error),
    )


if __name__ == "__main__":  # pragma: no cover - helper script
    main()
