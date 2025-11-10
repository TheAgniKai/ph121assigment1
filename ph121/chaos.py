"""Helpers for studying chaotic attractors.

This module provides a light-weight implementation of the workflow described in the
assignment for estimating the maximal Lyapunov exponent of a chaotic attractor.  We
focus on the classic Lorenz '63 system as a representative example because it is
well-known, low dimensional, and exhibits sustained chaotic dynamics for the
canonical choice of parameters (:math:`\sigma = 10`, :math:`\rho = 28`,
:math:`\beta = 8/3`).

The functions below rely solely on the standard library and the
``ph121.integrators`` helpers so that they can be executed on minimal
installations such as the autograder environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from random import Random
from statistics import mean, stdev
from typing import Iterable, List, Sequence, Tuple

from .integrators import IntegrationResult, runge_kutta4

# --- Lorenz attractor definition -------------------------------------------------


def lorenz63_derivative(
    _: float, state: Tuple[float, float, float], *, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0
) -> Tuple[float, float, float]:
    """Return the time derivative for the Lorenz '63 system.

    Parameters
    ----------
    state:
        Current point ``(x, y, z)`` in phase space.
    sigma, rho, beta:
        Standard Lorenz parameters.  The defaults correspond to the canonical
        chaotic regime.
    """

    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z


# --- Utilities -------------------------------------------------------------------


def _sample_gaussian_vector(dim: int, rng: Random) -> Tuple[float, ...]:
    """Draw a vector with independent standard normal components."""

    return tuple(rng.gauss(0.0, 1.0) for _ in range(dim))


def _normalise(vector: Sequence[float]) -> Tuple[float, ...]:
    """Return the unit vector in the direction of ``vector``."""

    norm_sq = sum(component * component for component in vector)
    if norm_sq == 0.0:
        raise ValueError("Cannot normalise the zero vector.")
    norm = sqrt(norm_sq)
    return tuple(component / norm for component in vector)


def _vector_add(a: Sequence[float], b: Sequence[float]) -> Tuple[float, ...]:
    return tuple(x + y for x, y in zip(a, b))


def _vector_sub(a: Sequence[float], b: Sequence[float]) -> Tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def _vector_scale(vector: Sequence[float], factor: float) -> Tuple[float, ...]:
    return tuple(factor * component for component in vector)


def _vector_norm(vector: Sequence[float]) -> float:
    return sqrt(sum(component * component for component in vector))


def _linear_regression_slope(points: Iterable[Tuple[float, float]]) -> float:
    """Compute the slope of a least squares fit to ``points``.

    The helper returns the slope of the linear fit ``y = a * x + b``.  We do not
    need to explicitly compute the intercept ``b`` in the Lyapunov estimator.
    """

    xs: List[float] = []
    ys: List[float] = []
    for x, y in points:
        xs.append(x)
        ys.append(y)

    if len(xs) < 2:
        raise ValueError("At least two points are required to fit a line.")

    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0.0:
        raise ValueError("Cannot fit a line to points with zero variance in x.")
    return numerator / denominator


# --- Lyapunov exponent estimation ------------------------------------------------


@dataclass(frozen=True)
class LyapunovEstimate:
    """Summary of a maximal Lyapunov exponent measurement."""

    exponent: float
    warmup_result: IntegrationResult
    base_result: IntegrationResult
    perturbed_result: IntegrationResult


def _filter_growth_region(
    times: Sequence[float],
    displacements: Sequence[float],
    *,
    initial_magnitude: float,
    lower_factor: float,
    upper_bound: float,
) -> List[Tuple[float, float]]:
    """Select the time window displaying exponential separation.

    ``lower_factor`` specifies the minimum allowed growth relative to the
    initial perturbation ``initial_magnitude``.  ``upper_bound`` is an absolute
    upper bound on the displacement magnitude to avoid the saturation region
    where linearisation breaks down.
    """

    points: List[Tuple[float, float]] = []
    for t, displacement in zip(times, displacements):
        if displacement <= 0.0:
            continue
        if displacement < lower_factor * initial_magnitude:
            continue
        if displacement > upper_bound:
            continue
        growth = displacement / initial_magnitude
        points.append((t, log(growth)))
    return points


def _integrate_lorenz(
    initial_state: Tuple[float, float, float],
    *,
    t_end: float,
    step: float,
) -> IntegrationResult:
    return runge_kutta4(lorenz63_derivative, initial_state, t0=0.0, t_end=t_end, step=step)


def estimate_lorenz_maximal_lyapunov(
    *,
    samples: int = 10,
    warmup_time: float = 20.0,
    evaluation_time: float = 8.0,
    step: float = 0.01,
    perturbation_magnitude: float = 1.0e-8,
    lower_growth_factor: float = 10.0,
    upper_displacement: float = 0.1,
    rng: Random | None = None,
) -> Tuple[List[LyapunovEstimate], float, float]:
    """Estimate the maximal Lyapunov exponent for the Lorenz attractor.

    Parameters
    ----------
    samples:
        Number of independent initial conditions to draw.
    warmup_time:
        Duration used to drive each initial condition onto the attractor.
    evaluation_time:
        Duration over which the exponential separation is measured.
    step:
        Integration time step.
    perturbation_magnitude:
        Initial distance between the reference and perturbed trajectories.
    lower_growth_factor:
        Minimum allowed growth relative to the initial perturbation before the
        displacement is considered in the exponential regime.  Values around 10
        effectively skip the initial transient.
    upper_displacement:
        Maximum allowed displacement before the separation saturates due to the
        finite size of the attractor.
    rng:
        Optional random number generator used to draw the initial conditions and
        perturbations.  If omitted a fresh :class:`random.Random` instance is
        created.

    Returns
    -------
    estimates, mean, standard_deviation:
        The detailed estimates for each sample together with the mean and sample
        standard deviation of the fitted Lyapunov exponents.
    """

    if samples <= 0:
        raise ValueError("At least one sample is required to estimate the exponent.")
    if perturbation_magnitude <= 0.0:
        raise ValueError("The perturbation magnitude must be positive.")
    if lower_growth_factor <= 1.0:
        raise ValueError("The lower growth factor should exceed unity to skip transients.")

    if rng is None:
        rng = Random()

    estimates: List[LyapunovEstimate] = []
    exponents: List[float] = []

    for _ in range(samples):
        # Draw a random initial condition near the origin and let it relax onto the attractor.
        raw_state = _sample_gaussian_vector(3, rng)
        warmup = _integrate_lorenz(raw_state, t_end=warmup_time, step=step)
        base_state = warmup.states[-1]

        # Generate a small random perturbation with Gaussian-distributed direction.
        perturbation_direction = _normalise(_sample_gaussian_vector(3, rng))
        perturbation = _vector_scale(perturbation_direction, perturbation_magnitude)
        perturbed_state = _vector_add(base_state, perturbation)

        # Integrate both trajectories from the same time origin.
        base = _integrate_lorenz(base_state, t_end=evaluation_time, step=step)
        perturbed = _integrate_lorenz(perturbed_state, t_end=evaluation_time, step=step)

        displacements = [_vector_norm(_vector_sub(p_state, b_state)) for b_state, p_state in zip(base.states, perturbed.states)]
        points = _filter_growth_region(
            base.times,
            displacements,
            initial_magnitude=perturbation_magnitude,
            lower_factor=lower_growth_factor,
            upper_bound=upper_displacement,
        )

        if len(points) < 2:
            raise RuntimeError(
                "Unable to identify a growth window for the Lyapunov fit. "
                "Consider increasing `evaluation_time` or adjusting the bounds."
            )

        slope = _linear_regression_slope(points)
        exponent = slope

        estimates.append(
            LyapunovEstimate(
                exponent=exponent,
                warmup_result=warmup,
                base_result=base,
                perturbed_result=perturbed,
            )
        )
        exponents.append(exponent)

    exponent_mean = mean(exponents)
    exponent_stdev = stdev(exponents) if len(exponents) > 1 else 0.0

    return estimates, exponent_mean, exponent_stdev


def format_estimates(estimates: Sequence[LyapunovEstimate]) -> str:
    """Return a human-readable summary table for a collection of estimates."""

    lines = ["Sample  λ_max [1/s]"]
    for index, estimate in enumerate(estimates, start=1):
        lines.append(f"{index:>6d}  {estimate.exponent:8.5f}")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - convenience entry point for manual runs
    estimates, exponent_mean, exponent_stdev = estimate_lorenz_maximal_lyapunov(samples=5)
    print(format_estimates(estimates))
    print()
    print(f"Mean λ_max = {exponent_mean:.5f} ± {exponent_stdev:.5f} 1/s")


if __name__ == "__main__":  # pragma: no cover - manual exploration helper
    main()

