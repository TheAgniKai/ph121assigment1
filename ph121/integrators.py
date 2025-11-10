"""Simple ODE solvers used in the PH121 assignments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

State = Tuple[float, ...]
DerivativeFunc = Callable[[float, Sequence[float]], Sequence[float]]
AccelerationFunc = Callable[[float, Sequence[float]], Sequence[float]]


@dataclass
class IntegrationResult:
    """Hold the sampled times and states."""

    times: Tuple[float, ...]
    states: Tuple[State, ...]

    def as_lists(self) -> Tuple[List[float], List[List[float]]]:
        times_list: List[float] = []
        for value in self.times:
            times_list.append(value)
        states_list: List[List[float]] = []
        for state in self.states:
            copy: List[float] = []
            for component in state:
                copy.append(component)
            states_list.append(copy)
        return times_list, states_list


def _to_state(values: Iterable[float]) -> State:
    parts: List[float] = []
    for entry in values:
        parts.append(float(entry))
    return tuple(parts)


def _time_grid(t0: float, t_end: float, step: float) -> Tuple[float, ...]:
    if step <= 0.0:
        raise ValueError("step must be positive")
    if t_end < t0:
        raise ValueError("t_end must be >= t0")
    points: List[float] = []
    current = float(t0)
    while current < t_end:
        points.append(current)
        next_value = current + step
        if next_value > t_end:
            next_value = t_end
        current = next_value
    if not points or points[-1] != t_end:
        points.append(t_end)
    return tuple(points)


def forward_euler(
    derivative: DerivativeFunc,
    initial_state: Sequence[float],
    *,
    t0: float,
    t_end: float,
    step: float,
) -> IntegrationResult:
    """Plain Euler integration."""

    times = _time_grid(t0, t_end, step)
    states: List[State] = []
    states.append(_to_state(initial_state))
    index = 0
    while index < len(times) - 1:
        t = times[index]
        dt = times[index + 1] - t
        state = states[index]
        slope = derivative(t, state)
        updated: List[float] = []
        for value, rate in zip(state, slope):
            updated.append(value + dt * float(rate))
        states.append(_to_state(updated))
        index += 1
    return IntegrationResult(times=times, states=tuple(states))


def runge_kutta4(
    derivative: DerivativeFunc,
    initial_state: Sequence[float],
    *,
    t0: float,
    t_end: float,
    step: float,
) -> IntegrationResult:
    """Classic fourth order RK integrator."""

    times = _time_grid(t0, t_end, step)
    states: List[State] = []
    states.append(_to_state(initial_state))
    index = 0
    while index < len(times) - 1:
        t = times[index]
        h = times[index + 1] - t
        y = states[index]
        k1 = tuple(float(value) for value in derivative(t, y))
        mid1: List[float] = []
        for value, slope in zip(y, k1):
            mid1.append(value + 0.5 * h * slope)
        y_k2 = _to_state(mid1)
        k2 = tuple(float(value) for value in derivative(t + 0.5 * h, y_k2))
        mid2: List[float] = []
        for value, slope in zip(y, k2):
            mid2.append(value + 0.5 * h * slope)
        y_k3 = _to_state(mid2)
        k3 = tuple(float(value) for value in derivative(t + 0.5 * h, y_k3))
        mid3: List[float] = []
        for value, slope in zip(y, k3):
            mid3.append(value + h * slope)
        y_k4 = _to_state(mid3)
        k4 = tuple(float(value) for value in derivative(t + h, y_k4))
        combined: List[float] = []
        for value, a, b, c, d in zip(y, k1, k2, k3, k4):
            piece = a + 2.0 * b + 2.0 * c + d
            combined.append(value + (h / 6.0) * piece)
        states.append(_to_state(combined))
        index += 1
    return IntegrationResult(times=times, states=tuple(states))


def velocity_verlet(
    acceleration: AccelerationFunc,
    initial_position: Sequence[float],
    initial_velocity: Sequence[float],
    *,
    t0: float,
    t_end: float,
    step: float,
) -> IntegrationResult:
    """Velocity Verlet stepper."""

    times = _time_grid(t0, t_end, step)
    position = _to_state(initial_position)
    velocity = _to_state(initial_velocity)
    acc = tuple(float(value) for value in acceleration(t0, position))
    states: List[State] = []
    states.append(position + velocity)
    index = 0
    while index < len(times) - 1:
        t = times[index]
        h = times[index + 1] - t
        half_step: List[float] = []
        for vel, a in zip(velocity, acc):
            half_step.append(vel + 0.5 * h * a)
        velocity_half = _to_state(half_step)
        new_pos_parts: List[float] = []
        for pos, vel_half in zip(position, velocity_half):
            new_pos_parts.append(pos + h * vel_half)
        position = _to_state(new_pos_parts)
        next_acc = tuple(float(value) for value in acceleration(t + h, position))
        new_vel_parts: List[float] = []
        for vel_half, a in zip(velocity_half, next_acc):
            new_vel_parts.append(vel_half + 0.5 * h * a)
        velocity = _to_state(new_vel_parts)
        acc = next_acc
        states.append(position + velocity)
        index += 1
    return IntegrationResult(times=times, states=tuple(states))


__all__ = [
    "IntegrationResult",
    "forward_euler",
    "runge_kutta4",
    "velocity_verlet",
]
