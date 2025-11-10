"""Small helpers for Lorenz attractor experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, log, sqrt
from random import Random
from statistics import mean, stdev
from typing import Dict, Iterable, List, Sequence, Tuple

from .integrators import IntegrationResult, runge_kutta4

# lorenz defaults
DEFAULT_SIGMA = 10.0
DEFAULT_RHO = 28.0
DEFAULT_BETA = 8.0 / 3.0


def lorenz63_derivative(
    _: float,
    state: Tuple[float, float, float],
    *,
    sigma: float = DEFAULT_SIGMA,
    rho: float = DEFAULT_RHO,
    beta: float = DEFAULT_BETA,
) -> Tuple[float, float, float]:
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz


def lorenz63_jacobian(
    state: Tuple[float, float, float],
    *,
    sigma: float = DEFAULT_SIGMA,
    rho: float = DEFAULT_RHO,
    beta: float = DEFAULT_BETA,
) -> Tuple[Tuple[float, float, float], ...]:
    x, y, z = state
    return (
        (-sigma, sigma, 0.0),
        (rho - z, -1.0, -x),
        (y, x, -beta),
    )


def _sample_gaussian_vector(dim: int, rng: Random) -> Tuple[float, ...]:
    values: List[float] = []
    for _ in range(dim):
        values.append(rng.gauss(0.0, 1.0))
    return tuple(values)


def _normalise(vector: Sequence[float]) -> Tuple[float, ...]:
    total = 0.0
    for component in vector:
        total += component * component
    if total == 0.0:
        raise ValueError("zero vector")
    length = sqrt(total)
    scaled: List[float] = []
    for component in vector:
        scaled.append(component / length)
    return tuple(scaled)


def _vector_add(a: Sequence[float], b: Sequence[float]) -> Tuple[float, ...]:
    result: List[float] = []
    for x, y in zip(a, b):
        result.append(x + y)
    return tuple(result)


def _vector_sub(a: Sequence[float], b: Sequence[float]) -> Tuple[float, ...]:
    result: List[float] = []
    for x, y in zip(a, b):
        result.append(x - y)
    return tuple(result)


def _vector_scale(vector: Sequence[float], factor: float) -> Tuple[float, ...]:
    result: List[float] = []
    for component in vector:
        result.append(component * factor)
    return tuple(result)


def _vector_norm(vector: Sequence[float]) -> float:
    total = 0.0
    for component in vector:
        total += component * component
    return sqrt(total)


def _vector_dot(a: Sequence[float], b: Sequence[float]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def _matrix_vector_product(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Tuple[float, ...]:
    result: List[float] = []
    for row in matrix:
        result.append(_vector_dot(row, vector))
    return tuple(result)


def _linear_regression_slope(points: Iterable[Tuple[float, float]]) -> float:
    xs: List[float] = []
    ys: List[float] = []
    for x, y in points:
        xs.append(x)
        ys.append(y)
    if len(xs) < 2:
        raise ValueError("need two points")
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = 0.0
    denominator = 0.0
    for x, y in zip(xs, ys):
        numerator += (x - x_mean) * (y - y_mean)
        denominator += (x - x_mean) ** 2
    if denominator == 0.0:
        raise ValueError("zero variance")
    return numerator / denominator


@dataclass(frozen=True)
class LyapunovEstimate:
    """Store one Lyapunov fit."""

    exponent: float
    warmup_result: IntegrationResult
    base_result: IntegrationResult
    perturbed_result: IntegrationResult


@dataclass(frozen=True)
class BoxCountingEstimate:
    """Store epsilon and visit count."""

    epsilon: float
    box_count: int


@dataclass(frozen=True)
class LyapunovSpectrumEstimate:
    """Store full spectrum and Kaplan–Yorke value."""

    exponents: Tuple[float, ...]
    kaplan_yorke_dimension: float


@dataclass(frozen=True)
class InformationDimensionEstimate:
    """Store epsilon and entropy."""

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
    pairs: List[Tuple[float, float]] = []
    for time, distance in zip(times, displacements):
        if distance <= 0.0:
            continue
        if distance < lower_factor * initial_magnitude:
            continue
        if distance > upper_bound:
            continue
        growth = distance / initial_magnitude
        pairs.append((time, log(growth)))
    return pairs


def _integrate_lorenz(
    initial_state: Tuple[float, float, float],
    *,
    t_end: float,
    step: float,
    sigma: float = DEFAULT_SIGMA,
    rho: float = DEFAULT_RHO,
    beta: float = DEFAULT_BETA,
) -> IntegrationResult:
    def _derivative(t: float, state: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return lorenz63_derivative(t, state, sigma=sigma, rho=rho, beta=beta)

    return runge_kutta4(_derivative, initial_state, t0=0.0, t_end=t_end, step=step)


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
    if samples <= 0:
        raise ValueError("samples must be positive")
    if perturbation_magnitude <= 0.0:
        raise ValueError("perturbation must be positive")
    if lower_growth_factor <= 1.0:
        raise ValueError("growth factor too small")
    if rng is None:
        rng = Random()
    estimates: List[LyapunovEstimate] = []
    exponents: List[float] = []
    for _ in range(samples):
        start = _sample_gaussian_vector(3, rng)
        warmup = _integrate_lorenz(start, t_end=warmup_time, step=step)
        base_state = warmup.states[-1]
        direction = _normalise(_sample_gaussian_vector(3, rng))
        perturbation = _vector_scale(direction, perturbation_magnitude)
        perturbed_state = _vector_add(base_state, perturbation)
        base = _integrate_lorenz(base_state, t_end=evaluation_time, step=step)
        perturbed = _integrate_lorenz(perturbed_state, t_end=evaluation_time, step=step)
        displacements: List[float] = []
        for base_state_step, perturbed_state_step in zip(base.states, perturbed.states):
            diff = _vector_sub(perturbed_state_step, base_state_step)
            displacements.append(_vector_norm(diff))
        points = _filter_growth_region(
            base.times,
            displacements,
            initial_magnitude=perturbation_magnitude,
            lower_factor=lower_growth_factor,
            upper_bound=upper_displacement,
        )
        if len(points) < 2:
            raise RuntimeError("no growth window")
        slope = _linear_regression_slope(points)
        estimates.append(
            LyapunovEstimate(
                exponent=slope,
                warmup_result=warmup,
                base_result=base,
                perturbed_result=perturbed,
            )
        )
        exponents.append(slope)
    exponent_mean = mean(exponents)
    exponent_stdev = stdev(exponents) if len(exponents) > 1 else 0.0
    return estimates, exponent_mean, exponent_stdev


def format_estimates(estimates: Sequence[LyapunovEstimate]) -> str:
    lines: List[str] = []
    lines.append("Sample  λ_max [1/s]")
    for index, estimate in enumerate(estimates, start=1):
        lines.append(f"{index:>6d}  {estimate.exponent:8.5f}")
    return "\n".join(lines)


def _bounding_box(states: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if not states:
        raise ValueError("need at least one state")
    dimensions = len(states[0])
    minima = [float("inf")] * dimensions
    maxima = [float("-inf")] * dimensions
    for state in states:
        if len(state) != dimensions:
            raise ValueError("dimension mismatch")
        for index, value in enumerate(state):
            if value < minima[index]:
                minima[index] = value
            if value > maxima[index]:
                maxima[index] = value
    padding = 1.0e-9
    for index in range(dimensions):
        span = maxima[index] - minima[index]
        if span == 0.0:
            span = 1.0
        offset = padding * span
        minima[index] -= offset
        maxima[index] += offset
    return tuple(minima), tuple(maxima)


def _count_visited_boxes(
    states: Sequence[Sequence[float]],
    epsilon: float,
    minima: Sequence[float],
    maxima: Sequence[float],
) -> int:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    dimensions = len(minima)
    bins_per_axis: List[int] = []
    for min_value, max_value in zip(minima, maxima):
        span = max_value - min_value
        if span <= 0.0:
            bins_per_axis.append(1)
        else:
            bins_per_axis.append(int(ceil(span / epsilon)) + 1)
    visited = set()
    for state in states:
        indices: List[int] = []
        for axis, value in enumerate(state):
            min_value = minima[axis]
            bins = bins_per_axis[axis]
            relative = (value - min_value) / epsilon
            index = floor(relative + 1.0e-12)
            if index < 0:
                index = 0
            if index >= bins:
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
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    dimensions = len(minima)
    bins_per_axis: List[int] = []
    for min_value, max_value in zip(minima, maxima):
        span = max_value - min_value
        if span <= 0.0:
            bins_per_axis.append(1)
        else:
            bins_per_axis.append(int(ceil(span / epsilon)) + 1)
    histogram: Dict[Tuple[int, ...], int] = {}
    for state in states:
        indices: List[int] = []
        for axis, value in enumerate(state):
            min_value = minima[axis]
            bins = bins_per_axis[axis]
            relative = (value - min_value) / epsilon
            index = floor(relative + 1.0e-12)
            if index < 0:
                index = 0
            if index >= bins:
                index = bins - 1
            indices.append(index)
        key = tuple(indices)
        current = histogram.get(key)
        if current is None:
            histogram[key] = 1
        else:
            histogram[key] = current + 1
    return histogram


def estimate_lorenz_box_counting_dimension(
    epsilons: Sequence[float],
    *,
    warmup_time: float = 20.0,
    sample_time: float = 20.0,
    step: float = 0.01,
    initial_state: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[List[BoxCountingEstimate], float]:
    if len(epsilons) < 2:
        raise ValueError("need two epsilons")
    if any(epsilon <= 0.0 for epsilon in epsilons):
        raise ValueError("epsilons must be positive")
    if sample_time <= 0.0:
        raise ValueError("sample_time must be positive")
    warmup = _integrate_lorenz(initial_state, t_end=warmup_time, step=step)
    sample = _integrate_lorenz(warmup.states[-1], t_end=sample_time, step=step)
    minima, maxima = _bounding_box(sample.states)
    sorted_eps = sorted(epsilons)
    estimates: List[BoxCountingEstimate] = []
    points: List[Tuple[float, float]] = []
    for epsilon in sorted_eps:
        count = _count_visited_boxes(sample.states, epsilon, minima, maxima)
        estimates.append(BoxCountingEstimate(epsilon=epsilon, box_count=count))
        if epsilon > 0.0 and count > 0:
            points.append((log(1.0 / epsilon), log(count)))
    if len(points) < 2:
        raise RuntimeError("not enough data")
    slope = _linear_regression_slope(points)
    return estimates, slope


def estimate_lorenz_information_dimension(
    epsilons: Sequence[float],
    *,
    warmup_time: float = 20.0,
    sample_time: float = 20.0,
    step: float = 0.01,
    finest_average: int = 4,
    initial_state: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[List[InformationDimensionEstimate], float]:
    if len(epsilons) < 2:
        raise ValueError("need two epsilons")
    if any(epsilon <= 0.0 for epsilon in epsilons):
        raise ValueError("epsilons must be positive")
    if sample_time <= 0.0:
        raise ValueError("sample_time must be positive")
    if finest_average <= 0:
        raise ValueError("finest_average must be positive")
    warmup = _integrate_lorenz(initial_state, t_end=warmup_time, step=step)
    sample = _integrate_lorenz(warmup.states[-1], t_end=sample_time, step=step)
    minima, maxima = _bounding_box(sample.states)
    sorted_eps = sorted(epsilons)
    total_points = len(sample.states)
    if total_points == 0:
        raise ValueError("no sample states")
    estimates: List[InformationDimensionEstimate] = []
    dimension_candidates: List[Tuple[float, float]] = []
    for epsilon in sorted_eps:
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
        raise RuntimeError("no candidates")
    fine_values = [value for epsilon, value in dimension_candidates if epsilon < 1.0]
    if len(fine_values) < finest_average:
        fine_values = [value for _, value in dimension_candidates]
    selected = sorted(fine_values)[: min(finest_average, len(fine_values))]
    dimension = sum(selected) / len(selected)
    return estimates, dimension


def _modified_gram_schmidt_with_norms(
    vectors: Sequence[Sequence[float]],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    orthonormal: List[Tuple[float, ...]] = []
    norms: List[float] = []
    for vector in vectors:
        work = [float(component) for component in vector]
        for basis_vector in orthonormal:
            projection = _vector_dot(work, basis_vector)
            for index, basis_component in enumerate(basis_vector):
                work[index] -= projection * basis_component
        norm_value = sqrt(sum(component * component for component in work))
        if norm_value == 0.0:
            raise ValueError("dependent basis")
        norms.append(norm_value)
        orthonormal.append(tuple(component / norm_value for component in work))
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
    if step <= 0.0:
        raise ValueError("step must be positive")
    state = tuple(float(component) for component in state)
    basis = tuple(tuple(float(component) for component in vector) for vector in basis)
    k1_state = lorenz63_derivative(0.0, state, sigma=sigma, rho=rho, beta=beta)
    jac_state = lorenz63_jacobian(state, sigma=sigma, rho=rho, beta=beta)
    k1_basis = tuple(_matrix_vector_product(jac_state, vector) for vector in basis)
    state_k2 = _vector_add(state, _vector_scale(k1_state, 0.5 * step))
    jac_k2 = lorenz63_jacobian(state_k2, sigma=sigma, rho=rho, beta=beta)
    basis_k2_input = []
    for vector, k1_vector in zip(basis, k1_basis):
        basis_k2_input.append(_vector_add(vector, _vector_scale(k1_vector, 0.5 * step)))
    k2_state = lorenz63_derivative(0.0, state_k2, sigma=sigma, rho=rho, beta=beta)
    k2_basis = tuple(_matrix_vector_product(jac_k2, vector) for vector in basis_k2_input)
    state_k3 = _vector_add(state, _vector_scale(k2_state, 0.5 * step))
    jac_k3 = lorenz63_jacobian(state_k3, sigma=sigma, rho=rho, beta=beta)
    basis_k3_input = []
    for vector, k2_vector in zip(basis, k2_basis):
        basis_k3_input.append(_vector_add(vector, _vector_scale(k2_vector, 0.5 * step)))
    k3_state = lorenz63_derivative(0.0, state_k3, sigma=sigma, rho=rho, beta=beta)
    k3_basis = tuple(_matrix_vector_product(jac_k3, vector) for vector in basis_k3_input)
    state_k4 = _vector_add(state, _vector_scale(k3_state, step))
    jac_k4 = lorenz63_jacobian(state_k4, sigma=sigma, rho=rho, beta=beta)
    basis_k4_input = []
    for vector, k3_vector in zip(basis, k3_basis):
        basis_k4_input.append(_vector_add(vector, _vector_scale(k3_vector, step)))
    k4_state = lorenz63_derivative(0.0, state_k4, sigma=sigma, rho=rho, beta=beta)
    k4_basis = tuple(_matrix_vector_product(jac_k4, vector) for vector in basis_k4_input)
    factor = step / 6.0
    next_state_parts: List[float] = []
    for value, a, b, c, d in zip(state, k1_state, k2_state, k3_state, k4_state):
        next_state_parts.append(value + factor * (a + 2.0 * b + 2.0 * c + d))
    next_state = tuple(next_state_parts)
    next_basis: List[Tuple[float, ...]] = []
    for vector, k1_vector, k2_vector, k3_vector, k4_vector in zip(basis, k1_basis, k2_basis, k3_basis, k4_basis):
        parts: List[float] = []
        for value, a, b, c, d in zip(vector, k1_vector, k2_vector, k3_vector, k4_vector):
            parts.append(value + factor * (a + 2.0 * b + 2.0 * c + d))
        next_basis.append(tuple(parts))
    return next_state, tuple(next_basis)


def estimate_lorenz_lyapunov_spectrum(
    *,
    warmup_time: float = 20.0,
    sample_time: float = 60.0,
    step: float = 0.01,
    initial_state: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    sigma: float = DEFAULT_SIGMA,
    rho: float = DEFAULT_RHO,
    beta: float = DEFAULT_BETA,
) -> LyapunovSpectrumEstimate:
    if sample_time <= 0.0:
        raise ValueError("sample_time must be positive")
    if step <= 0.0:
        raise ValueError("step must be positive")
    warmup = _integrate_lorenz(initial_state, t_end=warmup_time, step=step, sigma=sigma, rho=rho, beta=beta)
    state = warmup.states[-1]
    basis: List[Tuple[float, ...]] = []
    dimension = len(state)
    for index in range(dimension):
        vector = [0.0] * dimension
        vector[index] = 1.0
        basis.append(tuple(vector))
    exponent_sums = [0.0] * dimension
    elapsed = 0.0
    while elapsed < sample_time - 1.0e-12:
        h = min(step, sample_time - elapsed)
        state, propagated_basis = _advance_lorenz_and_tangent(state, basis, h, sigma=sigma, rho=rho, beta=beta)
        orthonormal_basis, norms = _modified_gram_schmidt_with_norms(propagated_basis)
        for index, norm_value in enumerate(norms):
            exponent_sums[index] += log(norm_value)
        basis = list(orthonormal_basis)
        elapsed += h
    if elapsed == 0.0:
        raise ValueError("no elapsed time")
    exponents = tuple(total / elapsed for total in exponent_sums)
    cumulative = 0.0
    k_index = -1
    partial_sums: List[float] = []
    for exponent in exponents:
        cumulative += exponent
        partial_sums.append(cumulative)
        if cumulative > 0.0:
            k_index = len(partial_sums) - 1
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


def main() -> None:  # pragma: no cover - helper
    estimates, exponent_mean, exponent_stdev = estimate_lorenz_maximal_lyapunov(samples=5)
    print(format_estimates(estimates))
    print()
    print(f"Mean λ_max = {exponent_mean:.5f} ± {exponent_stdev:.5f} 1/s")


if __name__ == "__main__":  # pragma: no cover - helper
    main()
