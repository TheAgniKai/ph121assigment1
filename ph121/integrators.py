"""Numerical integrators for ordinary differential equations.

The module implements a small collection of explicit schemes that are useful in the
PH121 numerical integration assignment:

* :func:`forward_euler` – first order explicit Euler integrator for general first-order systems.
* :func:`runge_kutta4` – classic fourth-order Runge–Kutta scheme.
* :func:`velocity_verlet` – symplectic velocity Verlet scheme for second-order systems.

All routines operate on callables that return the time-derivative of the state.  The
state is represented as a tuple of floating point numbers, keeping dependencies to the
Python standard library only.  This makes the helpers easy to reuse in environments
where third-party libraries such as NumPy are not available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

State = Tuple[float, ...]
DerivativeFunc = Callable[[float, State], Sequence[float]]
AccelerationFunc = Callable[[float, State], Sequence[float]]


@dataclass
class IntegrationResult:
    """Container holding the sampled time points and system states."""

    times: Tuple[float, ...]
    states: Tuple[State, ...]

    def as_lists(self) -> Tuple[List[float], List[List[float]]]:
        """Return copies of the time and state arrays as mutable lists."""

        return list(self.times), [list(state) for state in self.states]


def _ensure_state(state: Sequence[float]) -> State:
    """Convert *state* into an immutable :class:`State` tuple."""

    return tuple(float(component) for component in state)


def _advance_time_grid(t0: float, t_end: float, h: float) -> Tuple[float, ...]:
    if h <= 0:
        raise ValueError("Step size h must be positive.")
    if t_end < t0:
        raise ValueError("t_end must be greater than or equal to t0.")

    times: List[float] = []
    t = t0
    while t < t_end:
        times.append(t)
        t = min(t_end, t + h)
    if not times or times[-1] != t_end:
        times.append(t_end)
    return tuple(times)


def forward_euler(
    derivative: DerivativeFunc,
    initial_state: Sequence[float],
    *,
    t0: float,
    t_end: float,
    step: float,
) -> IntegrationResult:
    """Integrate a system of first-order equations using the forward Euler method.

    Parameters
    ----------
    derivative:
        Callable returning ``dy/dt`` given ``(t, y)``.
    initial_state:
        Iterable of floats describing the starting state ``y(t0)``.
    t0, t_end:
        Start and end times of the integration interval.
    step:
        Positive step size ``h``.

    Returns
    -------
    IntegrationResult
        Sampled time points (including ``t_end``) and states.
    """

    times = _advance_time_grid(t0, t_end, step)
    states: List[State] = [_ensure_state(initial_state)]

    for idx, t in enumerate(times[:-1]):
        dt = times[idx + 1] - t
        state = states[idx]
        slope = derivative(t, state)
        next_state = _ensure_state(
            component + dt * float(ds_dt)
            for component, ds_dt in zip(state, slope)
        )
        states.append(next_state)

    return IntegrationResult(times=times, states=tuple(states))


def runge_kutta4(
    derivative: DerivativeFunc,
    initial_state: Sequence[float],
    *,
    t0: float,
    t_end: float,
    step: float,
) -> IntegrationResult:
    """Integrate a system with the classic fourth-order Runge–Kutta scheme."""

    times = _advance_time_grid(t0, t_end, step)
    states: List[State] = [_ensure_state(initial_state)]

    for idx, t in enumerate(times[:-1]):
        h = times[idx + 1] - t
        y = states[idx]
        k1 = tuple(float(value) for value in derivative(t, y))
        y_k2 = _ensure_state(component + 0.5 * h * value for component, value in zip(y, k1))
        k2 = tuple(float(value) for value in derivative(t + 0.5 * h, y_k2))
        y_k3 = _ensure_state(component + 0.5 * h * value for component, value in zip(y, k2))
        k3 = tuple(float(value) for value in derivative(t + 0.5 * h, y_k3))
        y_k4 = _ensure_state(component + h * value for component, value in zip(y, k3))
        k4 = tuple(float(value) for value in derivative(t + h, y_k4))

        next_state = _ensure_state(
            component
            + (h / 6.0)
            * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i)
            for component, k1_i, k2_i, k3_i, k4_i in zip(y, k1, k2, k3, k4)
        )
        states.append(next_state)

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
    """Integrate a Newtonian system using the velocity Verlet scheme.

    The acceleration callable should return the second derivative of the position with
    respect to time, i.e. :math:`\ddot{y} = F(t, y)`.

    Returns an :class:`IntegrationResult` whose states contain the concatenated
    ``(position, velocity)`` vectors.
    """

    times = _advance_time_grid(t0, t_end, step)
    position = _ensure_state(initial_position)
    velocity = _ensure_state(initial_velocity)
    acc = tuple(float(value) for value in acceleration(t0, position))

    states: List[State] = [position + velocity]

    for idx, t in enumerate(times[:-1]):
        h = times[idx + 1] - t
        velocity_half = _ensure_state(v_i + 0.5 * h * a_i for v_i, a_i in zip(velocity, acc))
        position = _ensure_state(p_i + h * v_half for p_i, v_half in zip(position, velocity_half))
        next_acc = tuple(float(value) for value in acceleration(t + h, position))
        velocity = _ensure_state(
            v_half + 0.5 * h * a_i
            for v_half, a_i in zip(velocity_half, next_acc)
        )
        acc = next_acc
        states.append(position + velocity)

    return IntegrationResult(times=times, states=tuple(states))

