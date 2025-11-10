"""Basic Monte Carlo helpers with minimal dependencies."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

Number = float
Point = Tuple[Number, ...]


@dataclass
class MonteCarloResult:
    """Store value, error, sample count, and optional sample traces."""

    value: float
    error: float
    samples: int
    raw_samples: Tuple[float, ...] | None = None
    weighted_samples: Tuple[float, ...] | None = None


def monte_carlo_integrate(
    func: Callable[[Point], float],
    bounds: Sequence[Tuple[float, float]],
    samples: int,
    *,
    rng: random.Random | None = None,
) -> MonteCarloResult:
    """Uniform Monte Carlo estimate."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if rng is None:
        rng = random.Random()
    widths: List[float] = []
    for lower, upper in bounds:
        width = upper - lower
        if width <= 0.0:
            raise ValueError("bounds need upper > lower")
        widths.append(width)
    volume = 1.0
    for width in widths:
        volume *= width
    values: List[float] = []
    for _ in range(samples):
        coords_list: List[float] = []
        for (lower, _), width in zip(bounds, widths):
            coords_list.append(lower + rng.random() * width)
        point = tuple(coords_list)
        values.append(func(point))
    mean = sum(values) / samples
    if samples > 1:
        variance = sum((value - mean) ** 2 for value in values) / (samples - 1)
    else:
        variance = 0.0
    error = math.sqrt(variance / samples)
    weighted_values = tuple(volume * value for value in values)
    return MonteCarloResult(
        value=volume * mean,
        error=volume * error,
        samples=samples,
        raw_samples=tuple(values),
        weighted_samples=weighted_values,
    )


def importance_sampling(
    func: Callable[[Point], float],
    sampler: Callable[[random.Random], Point],
    weight: Callable[[Point], float],
    samples: int,
    *,
    rng: random.Random | None = None,
) -> MonteCarloResult:
    """Importance sampling estimate."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if rng is None:
        rng = random.Random()
    raw_values: List[float] = []
    contributions: List[float] = []
    for _ in range(samples):
        point = sampler(rng)
        raw = func(point)
        weighted = weight(point) * raw
        raw_values.append(raw)
        contributions.append(weighted)
    mean = sum(contributions) / samples
    if samples > 1:
        variance = sum((value - mean) ** 2 for value in contributions) / (samples - 1)
    else:
        variance = 0.0
    error = math.sqrt(variance / samples)
    return MonteCarloResult(
        value=mean,
        error=error,
        samples=samples,
        raw_samples=tuple(raw_values),
        weighted_samples=tuple(contributions),
    )


def breit_wigner_sampler(
    mass: float,
    width: float,
    *,
    rho_bounds: Tuple[float, float] | None = None,
) -> Callable[[random.Random], Tuple[float]]:
    """Sampler for the Breit–Wigner change of variables."""

    if width <= 0.0:
        raise ValueError("width must be positive")
    if mass <= 0.0:
        raise ValueError("mass must be positive")
    if rho_bounds is None:
        rho_min = -math.pi / 2.0
        rho_max = math.pi / 2.0
    else:
        rho_min, rho_max = rho_bounds
    if rho_max <= rho_min:
        raise ValueError("rho_max must exceed rho_min")
    scale = width * mass

    def _sampler(rng: random.Random) -> Tuple[float]:
        rho = rng.uniform(rho_min, rho_max)
        value = mass * mass + scale * math.tan(rho)
        return (value,)

    return _sampler


def breit_wigner_weight(
    mass: float,
    width: float,
    *,
    rho_bounds: Tuple[float, float] | None = None,
) -> Callable[[Point], float]:
    """Weight for :func:`breit_wigner_sampler`."""

    if width <= 0.0:
        raise ValueError("width must be positive")
    if mass <= 0.0:
        raise ValueError("mass must be positive")
    if rho_bounds is None:
        rho_min = -math.pi / 2.0
        rho_max = math.pi / 2.0
    else:
        rho_min, rho_max = rho_bounds
    if rho_max <= rho_min:
        raise ValueError("rho_max must exceed rho_min")
    rho_range = rho_max - rho_min
    scale = width * mass

    def _weight(point: Point) -> float:
        (m_squared,) = point
        denominator = (m_squared - mass * mass) ** 2 + scale * scale
        proposal = (scale / denominator) / rho_range
        if proposal <= 0.0:
            raise ValueError("proposal density must stay positive")
        return 1.0 / proposal

    return _weight


__all__ = [
    "MonteCarloResult",
    "breit_wigner_sampler",
    "breit_wigner_weight",
    "importance_sampling",
    "monte_carlo_integrate",
]
