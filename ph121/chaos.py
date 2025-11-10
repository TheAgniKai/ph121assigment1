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
from math import ceil, floor, log, sqrt
from random import Random
from statistics import mean, stdev
from typing import Dict, Iterable, List, Sequence, Tuple

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


def lorenz63_jacobian(
    state: Tuple[float, float, float], *, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0
) -> Tuple[Tuple[float, float, float], ...]:
    """Return the Jacobian matrix of the Lorenz '63 flow at ``state``."""

    x, y, z = state
    return (
        (-sigma, sigma, 0.0),
        (rho - z, -1.0, -x),
        (y, x, -beta),
    )


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


def _vector_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _matrix_vector_product(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Tuple[float, ...]:
    return tuple(_vector_dot(row, vector) for row in matrix)


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


@dataclass(frozen=True)
class BoxCountingEstimate:
    """Summary of the box-counting statistics for a single ``ϵ`` value."""

    epsilon: float
    box_count: int


@dataclass(frozen=True)
class LyapunovSpectrumEstimate:
    """Summary of the full Lyapunov spectrum and Kaplan–Yorke dimension."""

    exponents: Tuple[float, ...]
    kaplan_yorke_dimension: float


@dataclass(frozen=True)
class InformationDimensionEstimate:
    """Shannon entropy accumulated for a given ``ϵ`` scale."""

    epsilon: float
    entropy: float


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


def _bounding_box(states: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return the minima and maxima across ``states`` for each coordinate."""

    if not states:
        raise ValueError("At least one state is required to compute a bounding box.")

    dimensions = len(states[0])
    minima = [float("inf")] * dimensions
    maxima = [float("-inf")] * dimensions

    for state in states:
        if len(state) != dimensions:
            raise ValueError("All states must have the same dimensionality.")
        for idx, value in enumerate(state):
            if value < minima[idx]:
                minima[idx] = value
            if value > maxima[idx]:
                maxima[idx] = value

    # Expand the bounds by a tiny factor to protect against round-off spilling
    # points exactly on the upper boundary into an additional box when scaled.
    padding = 1.0e-9
    for idx in range(dimensions):
        span = maxima[idx] - minima[idx]
        if span == 0.0:
            span = 1.0
        offset = padding * span
        minima[idx] -= offset
        maxima[idx] += offset

    return tuple(minima), tuple(maxima)


def _count_visited_boxes(
    states: Sequence[Sequence[float]],
    epsilon: float,
    minima: Sequence[float],
    maxima: Sequence[float],
) -> int:
    """Return the number of ``ϵ``-boxes visited by ``states``."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive when counting boxes.")

    dimensions = len(minima)
    bins_per_axis = []
    for min_value, max_value in zip(minima, maxima):
        span = max_value - min_value
        if span <= 0.0:
            bins_per_axis.append(1)
        else:
            bins_per_axis.append(int(ceil(span / epsilon)) + 1)

    visited = set()
    for state in states:
        indices = []
        for axis, value in enumerate(state):
            min_value = minima[axis]
            bins = bins_per_axis[axis]
            relative = (value - min_value) / epsilon
            index = floor(relative + 1.0e-12)
            if index < 0:
                index = 0
            elif index >= bins:
                index = bins - 1
            indices.append(index)
        visited.add(tuple(indices))

    return len(visited)


def _box_histogram(
    states: Sequence[Sequence[float]],
    epsilon: float,
    minima: Sequence[float],
    maxima: Sequence[float],
) -> Dict[Tuple[int, ...], int]:
    """Return a histogram counting the visits to each ``ϵ``-box."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive when building a histogram.")

    dimensions = len(minima)
    bins_per_axis = []
    for min_value, max_value in zip(minima, maxima):
        span = max_value - min_value
        if span <= 0.0:
            bins_per_axis.append(1)
        else:
            bins_per_axis.append(int(ceil(span / epsilon)) + 1)

    histogram: Dict[Tuple[int, ...], int] = {}
    for state in states:
        indices = []
        for axis, value in enumerate(state):
            min_value = minima[axis]
            bins = bins_per_axis[axis]
            relative = (value - min_value) / epsilon
            index = floor(relative + 1.0e-12)
            if index < 0:
                index = 0
            elif index >= bins:
                index = bins - 1
            indices.append(index)
        key = tuple(indices)
        histogram[key] = histogram.get(key, 0) + 1

    return histogram


def _modified_gram_schmidt_with_norms(
    vectors: Sequence[Sequence[float]],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return an orthonormal basis together with the norms removed in each step."""

    orthonormal: List[Tuple[float, ...]] = []
    norms: List[float] = []
    for vector in vectors:
        work = [float(component) for component in vector]
        for basis_vector in orthonormal:
            projection = _vector_dot(work, basis_vector)
            for idx, basis_component in enumerate(basis_vector):
                work[idx] -= projection * basis_component
        norm = sqrt(sum(component * component for component in work))
        if norm == 0.0:
            raise ValueError("Encountered a linearly dependent set during Gram–Schmidt orthogonalisation.")
        norms.append(norm)
        orthonormal.append(tuple(component / norm for component in work))
    return tuple(orthonormal), tuple(norms)


def _advance_lorenz_and_tangent(
    state: Sequence[float],
    basis: Sequence[Sequence[float]],
    step: float,
    *,
    sigma: float,
    rho: float,
    beta: float,
) -> Tuple[Tuple[float, ...], Tuple[Tuple[float, ...], ...]]:
    """Advance the Lorenz system and tangent basis by a single RK4 step."""

    if step <= 0.0:
        raise ValueError("Step size must be positive when advancing the Lorenz system.")

    state = tuple(float(component) for component in state)
    basis = tuple(tuple(float(component) for component in vector) for vector in basis)

    # Stage 1
    k1_state = lorenz63_derivative(0.0, state, sigma=sigma, rho=rho, beta=beta)
    jac_state = lorenz63_jacobian(state, sigma=sigma, rho=rho, beta=beta)
    k1_basis = tuple(_matrix_vector_product(jac_state, vector) for vector in basis)

    # Stage 2
    state_k2 = _vector_add(state, _vector_scale(k1_state, 0.5 * step))
    jac_k2 = lorenz63_jacobian(state_k2, sigma=sigma, rho=rho, beta=beta)
    basis_k2_input = tuple(
        _vector_add(vector, _vector_scale(k1_vector, 0.5 * step))
        for vector, k1_vector in zip(basis, k1_basis)
    )
    k2_state = lorenz63_derivative(0.0, state_k2, sigma=sigma, rho=rho, beta=beta)
    k2_basis = tuple(_matrix_vector_product(jac_k2, vector) for vector in basis_k2_input)

    # Stage 3
    state_k3 = _vector_add(state, _vector_scale(k2_state, 0.5 * step))
    jac_k3 = lorenz63_jacobian(state_k3, sigma=sigma, rho=rho, beta=beta)
    basis_k3_input = tuple(
        _vector_add(vector, _vector_scale(k2_vector, 0.5 * step))
        for vector, k2_vector in zip(basis, k2_basis)
    )
    k3_state = lorenz63_derivative(0.0, state_k3, sigma=sigma, rho=rho, beta=beta)
    k3_basis = tuple(_matrix_vector_product(jac_k3, vector) for vector in basis_k3_input)

    # Stage 4
    state_k4 = _vector_add(state, _vector_scale(k3_state, step))
    jac_k4 = lorenz63_jacobian(state_k4, sigma=sigma, rho=rho, beta=beta)
    basis_k4_input = tuple(
        _vector_add(vector, _vector_scale(k3_vector, step))
        for vector, k3_vector in zip(basis, k3_basis)
    )
    k4_state = lorenz63_derivative(0.0, state_k4, sigma=sigma, rho=rho, beta=beta)
    k4_basis = tuple(_matrix_vector_product(jac_k4, vector) for vector in basis_k4_input)

    factor = step / 6.0
    next_state = tuple(
        component
        + factor * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i)
        for component, k1_i, k2_i, k3_i, k4_i in zip(state, k1_state, k2_state, k3_state, k4_state)
    )

    next_basis = tuple(
        tuple(
            component
            + factor * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i)
            for component, k1_i, k2_i, k3_i, k4_i in zip(vector, k1_vector, k2_vector, k3_vector, k4_vector)
        )
        for vector, k1_vector, k2_vector, k3_vector, k4_vector in zip(basis, k1_basis, k2_basis, k3_basis, k4_basis)
    )

    return next_state, next_basis


def estimate_lorenz_box_counting_dimension(
    epsilons: Sequence[float],
    *,
    warmup_time: float = 20.0,
    sample_time: float = 60.0,
    step: float = 0.01,
    initial_state: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[List[BoxCountingEstimate], float]:
    """Estimate the box-counting dimension ``D₀`` of the Lorenz attractor.

    The routine follows the description given in the assignment text.  It first
    integrates the Lorenz system for ``warmup_time`` to drive the initial
    condition onto the attractor.  A subsequent integration of duration
    ``sample_time`` traces out a trajectory whose visited ``ϵ``-boxes are
    counted for each ``ϵ`` in ``epsilons``.

    Parameters
    ----------
    epsilons:
        Iterable of box edge lengths to analyse.  The collection must contain at
        least two positive values to allow a linear fit of ``n(ϵ)`` against
        ``e(ϵ)``.
    warmup_time:
        Integration time used to approach the attractor.
    sample_time:
        Duration of the integration used for the box counting.
    step:
        Integration step size ``h`` used for both warmup and sampling runs.
    initial_state:
        Starting point of the integration.

    Returns
    -------
    estimates, dimension:
        The individual :class:`BoxCountingEstimate` objects for each ``ϵ`` and
        the slope obtained from a least-squares fit of
        ``n(ϵ) = ln N(ϵ)`` versus ``e(ϵ) = |ln ϵ|``.
    """

    epsilons = list(epsilons)
    if len(epsilons) < 2:
        raise ValueError("At least two epsilon values are required to fit a slope.")
    if any(epsilon <= 0.0 for epsilon in epsilons):
        raise ValueError("All epsilon values must be positive.")

    warmup = _integrate_lorenz(initial_state, t_end=warmup_time, step=step)
    sample = _integrate_lorenz(warmup.states[-1], t_end=sample_time, step=step)

    minima, maxima = _bounding_box(sample.states)

    sorted_epsilons = sorted(epsilons)
    estimates: List[BoxCountingEstimate] = []
    for epsilon in sorted_epsilons:
        count = _count_visited_boxes(sample.states, epsilon, minima, maxima)
        estimates.append(BoxCountingEstimate(epsilon=epsilon, box_count=count))

    regression_points = [(abs(log(estimate.epsilon)), log(estimate.box_count)) for estimate in estimates]
    dimension = _linear_regression_slope(regression_points)

    return estimates, dimension


def estimate_lorenz_information_dimension(
    epsilons: Sequence[float],
    *,
    warmup_time: float = 20.0,
    sample_time: float = 60.0,
    step: float = 0.01,
    initial_state: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    finest_average: int = 1,
) -> Tuple[List[InformationDimensionEstimate], float]:
    """Estimate the information dimension ``D₁`` of the Lorenz attractor."""

    epsilons = list(epsilons)
    if len(epsilons) < 2:
        raise ValueError("At least two epsilon values are required to estimate the information dimension.")
    if any(epsilon <= 0.0 for epsilon in epsilons):
        raise ValueError("All epsilon values must be positive when estimating the information dimension.")
    if sample_time <= 0.0:
        raise ValueError("sample_time must be positive when estimating the information dimension.")
    if finest_average <= 0:
        raise ValueError("finest_average must be positive when estimating the information dimension.")

    warmup = _integrate_lorenz(initial_state, t_end=warmup_time, step=step)
    sample = _integrate_lorenz(warmup.states[-1], t_end=sample_time, step=step)

    minima, maxima = _bounding_box(sample.states)

    sorted_epsilons = sorted(epsilons)
    estimates: List[InformationDimensionEstimate] = []
    total_points = len(sample.states)
    if total_points == 0:
        raise ValueError("Sampling produced no states for the information dimension estimate.")

    dimension_candidates: List[Tuple[float, float]] = []
    for epsilon in sorted_epsilons:
        histogram = _box_histogram(sample.states, epsilon, minima, maxima)
        entropy = 0.0
        for count in histogram.values():
            probability = count / total_points
            if probability > 0.0:
                entropy -= probability * log(probability)
        estimates.append(InformationDimensionEstimate(epsilon=epsilon, entropy=entropy))
        scale = abs(log(epsilon))
        if scale > 0.0:
            dimension_candidates.append((epsilon, entropy / scale))

    if not dimension_candidates:
        raise RuntimeError("Unable to compute any information-dimension candidates.")

    fine_scale_values = [value for epsilon, value in dimension_candidates if epsilon < 1.0]
    if len(fine_scale_values) < finest_average:
        fine_scale_values = [value for _, value in dimension_candidates]

    selected = sorted(fine_scale_values)[: min(finest_average, len(fine_scale_values))]
    dimension = sum(selected) / len(selected)

    return estimates, dimension


def estimate_lorenz_lyapunov_spectrum(
    *,
    warmup_time: float = 20.0,
    sample_time: float = 60.0,
    step: float = 0.01,
    initial_state: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> LyapunovSpectrumEstimate:
    """Estimate the full Lyapunov spectrum and Kaplan–Yorke dimension."""

    if sample_time <= 0.0:
        raise ValueError("sample_time must be positive when estimating the Lyapunov spectrum.")
    if step <= 0.0:
        raise ValueError("step must be positive when estimating the Lyapunov spectrum.")

    warmup = _integrate_lorenz(initial_state, t_end=warmup_time, step=step)
    state = warmup.states[-1]

    basis = []
    dimension = len(state)
    for idx in range(dimension):
        vector = [0.0] * dimension
        vector[idx] = 1.0
        basis.append(tuple(vector))

    exponent_sums = [0.0] * dimension
    elapsed = 0.0

    while elapsed < sample_time - 1.0e-12:
        h = min(step, sample_time - elapsed)
        state, propagated_basis = _advance_lorenz_and_tangent(state, basis, h, sigma=sigma, rho=rho, beta=beta)
        orthonormal_basis, norms = _modified_gram_schmidt_with_norms(propagated_basis)
        for idx, norm in enumerate(norms):
            exponent_sums[idx] += log(norm)
        basis = list(orthonormal_basis)
        elapsed += h

    if elapsed == 0.0:
        raise ValueError("No time elapsed during the Lyapunov spectrum estimate.")

    exponents = tuple(sum_value / elapsed for sum_value in exponent_sums)

    cumulative = 0.0
    k_index = -1
    partial_sums: List[float] = []
    for idx, exponent in enumerate(exponents):
        cumulative += exponent
        partial_sums.append(cumulative)
        if cumulative > 0.0:
            k_index = idx
        else:
            break

    if k_index == -1:
        kaplan_yorke_dimension = 0.0
    elif k_index + 1 >= len(exponents):
        kaplan_yorke_dimension = float(len(exponents))
    else:
        numerator = partial_sums[k_index]
        next_exponent = exponents[k_index + 1]
        if next_exponent == 0.0:
            kaplan_yorke_dimension = float(k_index + 1)
        else:
            kaplan_yorke_dimension = (k_index + 1) + numerator / abs(next_exponent)

    return LyapunovSpectrumEstimate(exponents=exponents, kaplan_yorke_dimension=kaplan_yorke_dimension)


def main() -> None:  # pragma: no cover - convenience entry point for manual runs
    estimates, exponent_mean, exponent_stdev = estimate_lorenz_maximal_lyapunov(samples=5)
    print(format_estimates(estimates))
    print()
    print(f"Mean λ_max = {exponent_mean:.5f} ± {exponent_stdev:.5f} 1/s")


if __name__ == "__main__":  # pragma: no cover - manual exploration helper
    main()

