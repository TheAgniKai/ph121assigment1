"""Numerical helpers for the high energy Monte Carlo tasks."""

from __future__ import annotations

import math
import random
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .monte_carlo import (
    MonteCarloResult,
    breit_wigner_sampler,
    breit_wigner_weight,
    importance_sampling,
    monte_carlo_integrate,
)
from .pdfs import PartonDistribution, make_pdf

# basic constants
ALPHA_EM = 1.0 / 128.0
SIN2_THETA_W = 0.2312
COS2_THETA_W = 1.0 - SIN2_THETA_W
MASS_Z = 91.1876
WIDTH_Z = 2.4952
GEV2_TO_PB = 0.389379338e9
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

QUARK_VECTOR: Dict[int, float] = {}
for flavour in QUARK_FLAVOURS:
    QUARK_VECTOR[flavour] = _vector_coupling(QUARK_T3[flavour], QUARK_CHARGES[flavour])

QUARK_AXIAL: Dict[int, float] = {}
for flavour in QUARK_FLAVOURS:
    QUARK_AXIAL[flavour] = _axial_coupling(QUARK_T3[flavour])


def differential_cross_section_electron_muon(cos_theta: float, s: float) -> float:
    return (ALPHA_EM**2 / (4.0 * s)) * (1.0 + cos_theta * cos_theta)


def total_cross_section_electron_muon(
    *,
    energy: float,
    samples: int,
    rng=None,
) -> Tuple[MonteCarloResult, float]:
    s = energy * energy

    def _integrand(point: Tuple[float, float]) -> float:
        cos_value, _phi = point
        return differential_cross_section_electron_muon(cos_value, s)

    result = monte_carlo_integrate(
        _integrand,
        bounds=[(-1.0, 1.0), (0.0, 2.0 * math.pi)],
        samples=samples,
        rng=rng,
    )
    analytic = (4.0 * math.pi * ALPHA_EM**2) / (3.0 * s)
    return result, analytic


def _chi_components(s_hat: float) -> Tuple[float, float]:
    prefactor = s_hat / (4.0 * SIN2_THETA_W * COS2_THETA_W)
    delta = s_hat - MASS_Z * MASS_Z
    denom = delta * delta + (MASS_Z * WIDTH_Z) ** 2
    re_value = prefactor * delta / denom
    abs_sq = (prefactor * prefactor) / denom
    return re_value, abs_sq


def partonic_cross_section(s_hat: float, flavour: int) -> float:
    charge = QUARK_CHARGES[flavour]
    vector_q = QUARK_VECTOR[flavour]
    axial_q = QUARK_AXIAL[flavour]
    re_chi, abs_chi_sq = _chi_components(s_hat)
    coeff = charge * charge
    coeff -= 2.0 * charge * LEPTON_VECTOR * vector_q * re_chi
    coeff += (LEPTON_VECTOR**2 + LEPTON_AXIAL**2) * (vector_q**2 + axial_q**2) * abs_chi_sq
    return (4.0 * math.pi * ALPHA_EM**2 / (3.0 * s_hat)) * coeff


def _make_tau_sampler(s: float, tau_min: float):
    m2_min = tau_min * s
    m2_max = s
    rho_min = math.atan((m2_min - MASS_Z * MASS_Z) / (MASS_Z * WIDTH_Z))
    rho_max = math.atan((m2_max - MASS_Z * MASS_Z) / (MASS_Z * WIDTH_Z))
    sampler = breit_wigner_sampler(MASS_Z, WIDTH_Z, rho_bounds=(rho_min, rho_max))
    weight = breit_wigner_weight(MASS_Z, WIDTH_Z, rho_bounds=(rho_min, rho_max))
    y_max_global = -0.5 * math.log(tau_min)

    def _sampler(rng: random.Random) -> Tuple[float, float]:
        (m2,) = sampler(rng)
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
    y_max = -0.5 * math.log(max(tau, 1e-300))
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
    return total / s, tau


@dataclass
class DifferentialHistogram:
    """Store histogram values for dσ/dτ."""

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
        weight_value = weight(point)
        contribution = weight_value * value
        contributions.append(contribution)
        if contribution != 0.0:
            index = bisect_left(bin_edges, tau) - 1
            if 0 <= index < len(bin_sums) and bin_edges[index] <= tau < bin_edges[index + 1]:
                bin_sums[index] += contribution
                bin_sums_sq[index] += contribution * contribution
    mean = sum(contributions) / samples
    if samples > 1:
        variance = sum((value - mean) ** 2 for value in contributions) / (samples - 1)
    else:
        variance = 0.0
    total_result = MonteCarloResult(
        value=mean * GEV2_TO_PB,
        error=math.sqrt(variance / samples) * GEV2_TO_PB,
        samples=samples,
    )
    bin_values: List[float] = []
    bin_errors: List[float] = []
    for idx in range(len(bin_edges) - 1):
        width = bin_edges[idx + 1] - bin_edges[idx]
        if width <= 0.0:
            raise ValueError("Histogram bin edges must be strictly increasing")
        mean_bin = bin_sums[idx] / samples
        mean_sq_bin = bin_sums_sq[idx] / samples
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
    "MASS_Z",
    "WIDTH_Z",
    "differential_cross_section_electron_muon",
    "drell_yan_differential_cross_section",
    "drell_yan_total_cross_section",
    "partonic_cross_section",
    "total_cross_section_electron_muon",
]
