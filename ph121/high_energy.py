"""Monte Carlo integrations for high-energy scattering processes.

This module implements the optional Monte Carlo exercises described in the PH121
assignment: the total cross section for

``e^+ e^- -> gamma -> mu^+ mu^-``

and the Drell–Yan production of muon pairs at the LHC,

``p p -> gamma/Z -> mu^+ mu^-``.

Two integrations are required for the proton–proton case: the total cross
section and the differential cross section ``dσ/dτ`` extracted from the same
Monte Carlo samples.  The implementation follows the hints in the handout by
performing a change of variables that flattens the Breit–Wigner resonance of the
``γ/Z`` propagator.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from bisect import bisect_left
from typing import Dict, Iterable, List, Sequence, Tuple

from .monte_carlo import (
    MonteCarloResult,
    breit_wigner_sampler,
    breit_wigner_weight,
    importance_sampling,
    monte_carlo_integrate,
)
from .pdfs import PartonDistribution, make_pdf


# --- Constants -----------------------------------------------------------------

# Electroweak parameters (Particle Data Group 2024 values).
ALPHA_EM = 1.0 / 128.0
SIN2_THETA_W = 0.2312
COS2_THETA_W = 1.0 - SIN2_THETA_W
MASS_Z = 91.1876  # GeV
WIDTH_Z = 2.4952  # GeV

# Unit conversion: natural units (GeV^-2) to picobarns.
GEV2_TO_PB = 0.389379338e9

# Particle IDs for quarks in the LHAPDF scheme.  Positive values correspond to
# quarks, negative values to antiquarks.
QUARK_FLAVOURS: Tuple[int, ...] = (1, 2, 3, 4, 5)


def _vector_coupling(t3: float, charge: float) -> float:
    return t3 - 2.0 * charge * SIN2_THETA_W


def _axial_coupling(t3: float) -> float:
    return t3


LEPTON_CHARGE = -1.0
LEPTON_T3 = -0.5
LEPTON_VECTOR = _vector_coupling(LEPTON_T3, LEPTON_CHARGE)
LEPTON_AXIAL = _axial_coupling(LEPTON_T3)

QUARK_CHARGES: Dict[int, float] = {
    1: -1.0 / 3.0,
    2: 2.0 / 3.0,
    3: -1.0 / 3.0,
    4: 2.0 / 3.0,
    5: -1.0 / 3.0,
}

QUARK_T3: Dict[int, float] = {
    1: -0.5,
    2: 0.5,
    3: -0.5,
    4: 0.5,
    5: -0.5,
}

QUARK_VECTOR: Dict[int, float] = {
    flavour: _vector_coupling(QUARK_T3[flavour], QUARK_CHARGES[flavour])
    for flavour in QUARK_FLAVOURS
}
QUARK_AXIAL: Dict[int, float] = {
    flavour: _axial_coupling(QUARK_T3[flavour]) for flavour in QUARK_FLAVOURS
}


# --- Helper functions -----------------------------------------------------------


def differential_cross_section_electron_muon(cos_theta: float, s: float) -> float:
    """Return ``dσ/dΩ`` for ``e^+ e^- -> γ -> μ^+ μ^-``.

    Parameters
    ----------
    cos_theta:
        Cosine of the scattering angle.
    s:
        Centre-of-mass energy squared in GeV².
    """

    return (ALPHA_EM**2 / (4.0 * s)) * (1.0 + cos_theta * cos_theta)


def total_cross_section_electron_muon(
    *,
    energy: float,
    samples: int,
    rng=None,
) -> Tuple[MonteCarloResult, float]:
    """Integrate Eq. (18) of the assignment using Monte Carlo sampling.

    The function integrates ``dσ/dΩ`` over ``cos θ`` and ``φ`` to obtain the total
    cross section.  The analytic result is returned alongside the Monte Carlo
    estimate for comparison.
    """

    s = energy * energy

    def _integrand(point: Tuple[float, float]) -> float:
        cos_theta, _phi = point
        return differential_cross_section_electron_muon(cos_theta, s)

    result = monte_carlo_integrate(
        _integrand,
        bounds=[(-1.0, 1.0), (0.0, 2.0 * math.pi)],
        samples=samples,
        rng=rng,
    )

    analytic = (4.0 * math.pi * ALPHA_EM**2) / (3.0 * s)
    return result, analytic


def _chi_components(s_hat: float) -> Tuple[float, float]:
    """Return ``Re χ`` and ``|χ|²`` for the ``γ/Z`` propagator."""

    prefactor = s_hat / (4.0 * SIN2_THETA_W * COS2_THETA_W)
    delta = s_hat - MASS_Z**2
    denom = delta * delta + (MASS_Z * WIDTH_Z) ** 2
    re_chi = prefactor * delta / denom
    abs_chi_sq = (prefactor**2) / denom
    return re_chi, abs_chi_sq


def partonic_cross_section(s_hat: float, flavour: int) -> float:
    """Return the ``q \bar{q} -> μ^+ μ^-`` cross section integrated over cos θ."""

    charge = QUARK_CHARGES[flavour]
    vector_q = QUARK_VECTOR[flavour]
    axial_q = QUARK_AXIAL[flavour]

    re_chi, abs_chi_sq = _chi_components(s_hat)
    coeff_even = (
        charge * charge
        - 2.0 * charge * LEPTON_VECTOR * vector_q * re_chi
        + (LEPTON_VECTOR**2 + LEPTON_AXIAL**2) * (vector_q**2 + axial_q**2) * abs_chi_sq
    )
    return (4.0 * math.pi * ALPHA_EM**2 / (3.0 * s_hat)) * coeff_even


def _make_tau_sampler(s: float, tau_min: float):
    m2_min = tau_min * s
    m2_max = s

    rho_min = math.atan((m2_min - MASS_Z**2) / (MASS_Z * WIDTH_Z))
    rho_max = math.atan((m2_max - MASS_Z**2) / (MASS_Z * WIDTH_Z))
    sampler = breit_wigner_sampler(MASS_Z, WIDTH_Z, rho_bounds=(rho_min, rho_max))
    weight = breit_wigner_weight(MASS_Z, WIDTH_Z, rho_bounds=(rho_min, rho_max))
    y_max_global = -0.5 * math.log(tau_min)

    def _sampler(rng: random.Random) -> Tuple[float, float]:
        (m2,) = sampler(rng)
        # The assignment suggests sampling a wider rapidity range and simply
        # discarding events outside the physical region.  We therefore draw y
        # uniformly in the fixed interval associated with τ_min.
        y = rng.uniform(-y_max_global, y_max_global)
        return m2, y

    def _weight(point: Tuple[float, float]) -> float:
        (m2, _y) = point
        return weight((m2,)) * (2.0 * y_max_global)

    return _sampler, _weight


def _drell_yan_integrand(
    point: Tuple[float, float],
    *,
    s: float,
    tau_min: float,
    pdf: PartonDistribution,
    flavours: Sequence[int],
) -> Tuple[float, float]:
    m2, y = point
    tau = m2 / s
    if tau < tau_min or tau >= 1.0:
        return 0.0, tau

    y_max = -0.5 * math.log(tau)
    if abs(y) > y_max:
        return 0.0, tau

    sqrt_tau = math.sqrt(tau)
    x1 = sqrt_tau * math.exp(y)
    x2 = sqrt_tau * math.exp(-y)
    if x1 <= 0.0 or x1 >= 1.0 or x2 <= 0.0 or x2 >= 1.0:
        return 0.0, tau

    mu2 = max(m2, tau_min * s)
    total = 0.0
    for flavour in flavours:
        pdf_q1 = pdf.xfx(flavour, x1, mu2)
        pdf_q2 = pdf.xfx(flavour, x2, mu2)
        pdf_qbar1 = pdf.xfx(-flavour, x1, mu2)
        pdf_qbar2 = pdf.xfx(-flavour, x2, mu2)
        sigma_hat = partonic_cross_section(m2, flavour)
        total += (pdf_q1 * pdf_qbar2 + pdf_qbar1 * pdf_q2) * sigma_hat

    # Change of variables τ -> m² introduces an extra 1/s factor.
    return total / s, tau


@dataclass
class DifferentialHistogram:
    """Histogram of ``dσ/dτ`` values extracted from the Monte Carlo samples."""

    bin_edges: Sequence[float]
    bin_values: Sequence[float]
    bin_errors: Sequence[float]


def drell_yan_total_cross_section(
    *,
    energy: float,
    samples: int,
    q_min: float = 60.0,
    rng=None,
    pdf: PartonDistribution | None = None,
    flavours: Sequence[int] = QUARK_FLAVOURS,
) -> MonteCarloResult:
    """Monte Carlo estimate of ``σ(pp -> μ^+ μ^-)`` at the LHC."""

    if pdf is None:
        pdf = make_pdf()

    s = energy * energy
    tau_min = (q_min * q_min) / s
    sampler, weight = _make_tau_sampler(s, tau_min)

    def _integrand(point: Tuple[float, float]) -> float:
        value, _tau = _drell_yan_integrand(
            point,
            s=s,
            tau_min=tau_min,
            pdf=pdf,
            flavours=flavours,
        )
        return value

    result = importance_sampling(
        _integrand,
        sampler,
        weight,
        samples=samples,
        rng=rng,
    )

    return MonteCarloResult(
        value=result.value * GEV2_TO_PB,
        error=result.error * GEV2_TO_PB,
        samples=result.samples,
    )


def drell_yan_differential_cross_section(
    *,
    energy: float,
    samples: int,
    bins: Iterable[float],
    q_min: float = 60.0,
    rng=None,
    pdf: PartonDistribution | None = None,
    flavours: Sequence[int] = QUARK_FLAVOURS,
) -> Tuple[MonteCarloResult, DifferentialHistogram]:
    """Return the total and differential cross section ``dσ/dτ``."""

    if pdf is None:
        pdf = make_pdf()

    s = energy * energy
    tau_min = (q_min * q_min) / s
    sampler, weight = _make_tau_sampler(s, tau_min)

    bin_edges = list(bins)
    if sorted(bin_edges) != bin_edges:
        raise ValueError("Histogram bin edges must be sorted")
    if len(bin_edges) < 2:
        raise ValueError("At least two bin edges are required")

    bin_sums = [0.0 for _ in range(len(bin_edges) - 1)]
    bin_sums_sq = [0.0 for _ in range(len(bin_edges) - 1)]
    contributions: List[float] = []

    if rng is None:
        rng = random.Random()

    for _ in range(samples):
        point = sampler(rng)
        value, tau = _drell_yan_integrand(
            point,
            s=s,
            tau_min=tau_min,
            pdf=pdf,
            flavours=flavours,
        )
        w = weight(point)
        contribution = w * value
        contributions.append(contribution)

        if contribution != 0.0:
            idx = bisect_left(bin_edges, tau) - 1
            if 0 <= idx < len(bin_sums) and bin_edges[idx] <= tau < bin_edges[idx + 1]:
                bin_sums[idx] += contribution
                bin_sums_sq[idx] += contribution * contribution

    mean = sum(contributions) / samples
    variance = sum((c - mean) ** 2 for c in contributions) / (samples - 1 if samples > 1 else 1)
    total_result = MonteCarloResult(
        value=mean * GEV2_TO_PB,
        error=math.sqrt(variance / samples) * GEV2_TO_PB,
        samples=samples,
    )

    bin_values: List[float] = []
    bin_errors: List[float] = []
    for index in range(len(bin_edges) - 1):
        width = bin_edges[index + 1] - bin_edges[index]
        if width <= 0:
            raise ValueError("Histogram bin edges must be strictly increasing")
        mean_bin = bin_sums[index] / samples
        mean_sq_bin = bin_sums_sq[index] / samples
        variance_bin = max(mean_sq_bin - mean_bin * mean_bin, 0.0) / samples
        bin_values.append((mean_bin * GEV2_TO_PB) / width)
        bin_errors.append(math.sqrt(variance_bin) * GEV2_TO_PB / width)

    histogram = DifferentialHistogram(
        bin_edges=bin_edges,
        bin_values=bin_values,
        bin_errors=bin_errors,
    )

    return total_result, histogram


__all__ = [
    "DifferentialHistogram",
    "differential_cross_section_electron_muon",
    "drell_yan_differential_cross_section",
    "drell_yan_total_cross_section",
    "MASS_Z",
    "WIDTH_Z",
    "partonic_cross_section",
    "total_cross_section_electron_muon",
]

