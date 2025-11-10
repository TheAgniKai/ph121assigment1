"""Monte Carlo integration helpers used in the PH121 assignment."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

Number = float
Point = Tuple[Number, ...]


@dataclass
class MonteCarloResult:
    """Result of a Monte Carlo integration run."""

    value: float
    error: float
    samples: int


def monte_carlo_integrate(
    func: Callable[[Point], float],
    bounds: Sequence[Tuple[float, float]],
    samples: int,
    *,
    rng: random.Random | None = None,
) -> MonteCarloResult:
    """Estimate an integral using uniform Monte Carlo sampling.

    Parameters
    ----------
    func:
        Callable to evaluate. Receives a tuple containing the coordinates of a random
        sample inside the integration hyper-rectangle.
    bounds:
        Sequence of ``(min, max)`` tuples specifying the integration range in each
        dimension.
    samples:
        Number of random points used for the estimate. Must be positive.
    rng:
        Optional :class:`random.Random` instance controlling the pseudo-random stream.

    Returns
    -------
    MonteCarloResult
        Estimated integral value and its one-sigma statistical uncertainty.
    """

    if samples <= 0:
        raise ValueError("samples must be a positive integer")
    if rng is None:
        rng = random.Random()

    widths = [upper - lower for lower, upper in bounds]
    if any(width <= 0 for width in widths):
        raise ValueError("bounds must have upper > lower in every dimension")

    volume = math.prod(widths)
    evaluations: List[float] = []

    for _ in range(samples):
        coords = tuple(lower + rng.random() * width for (lower, _), width in zip(bounds, widths))
        evaluations.append(func(coords))

    mean = sum(evaluations) / samples
    variance = sum((value - mean) ** 2 for value in evaluations) / (samples - 1 if samples > 1 else 1)
    error = math.sqrt(variance / samples)
    return MonteCarloResult(value=volume * mean, error=volume * error, samples=samples)


def importance_sampling(
    func: Callable[[Point], float],
    sampler: Callable[[random.Random], Point],
    weight: Callable[[Point], float],
    samples: int,
    *,
    rng: random.Random | None = None,
) -> MonteCarloResult:
    """Estimate an integral using importance sampling.

    The sampler should draw points according to the proposal distribution, and the
    ``weight`` callable must return ``1 / g(x)`` where ``g`` is the probability density
    associated with the sampler.  The routine then averages ``f(x) / g(x)`` and returns
    the corresponding estimate of the integral along with the statistical error bar.
    """

    if samples <= 0:
        raise ValueError("samples must be a positive integer")
    if rng is None:
        rng = random.Random()

    evaluations: List[float] = []
    for _ in range(samples):
        point = sampler(rng)
        evaluations.append(weight(point) * func(point))

    mean = sum(evaluations) / samples
    variance = sum((value - mean) ** 2 for value in evaluations) / (samples - 1 if samples > 1 else 1)
    error = math.sqrt(variance / samples)
    return MonteCarloResult(value=mean, error=error, samples=samples)


def breit_wigner_sampler(
    mass: float,
    width: float,
    *,
    rho_bounds: Tuple[float, float] | None = None,
) -> Callable[[random.Random], Tuple[float]]:
    """Return a sampler that draws ``m^2`` values using the Breit–Wigner mapping."""

    if width <= 0:
        raise ValueError("width must be positive")
    if mass <= 0:
        raise ValueError("mass must be positive")

    if rho_bounds is None:
        rho_min, rho_max = -math.pi / 2.0, math.pi / 2.0
    else:
        rho_min, rho_max = rho_bounds
    if rho_max <= rho_min:
        raise ValueError("rho_bounds must satisfy rho_max > rho_min")
    scale = width * mass

    def _sampler(rng: random.Random) -> Tuple[float]:
        rho = rng.uniform(rho_min, rho_max)
        m_squared = mass**2 + scale * math.tan(rho)
        return (m_squared,)

    return _sampler


def breit_wigner_weight(
    mass: float,
    width: float,
    *,
    rho_bounds: Tuple[float, float] | None = None,
) -> Callable[[Point], float]:
    """Weight factor corresponding to :func:`breit_wigner_sampler`.

    The sampler draws points by choosing :math:`\rho \in [\rho_{\min}, \rho_{\max}]`
    uniformly and mapping it to ``m^2 = M\Gamma \tan \rho + M^2``.  The associated
    probability density in ``m^2`` therefore reads

    .. math::

        g(m^2) = \frac{1}{\rho_{\max} - \rho_{\min}} \frac{M\Gamma}{(m^2 - M^2)^2 + M^2\Gamma^2}.

    The returned callable evaluates ``1 / g(m^2)`` so that
    :func:`importance_sampling` can compute ``f(m^2) / g(m^2)``.
    """

    if width <= 0:
        raise ValueError("width must be positive")
    if mass <= 0:
        raise ValueError("mass must be positive")

    if rho_bounds is None:
        rho_min, rho_max = -math.pi / 2.0, math.pi / 2.0
    else:
        rho_min, rho_max = rho_bounds
    if rho_max <= rho_min:
        raise ValueError("rho_bounds must satisfy rho_max > rho_min")

    rho_range = rho_max - rho_min
    scale = width * mass

    def _weight(point: Point) -> float:
        (m_squared,) = point
        denominator = (m_squared - mass**2) ** 2 + scale**2
        proposal_density = (scale / denominator) / rho_range
        return 1.0 / proposal_density

    return _weight

