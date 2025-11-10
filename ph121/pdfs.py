"""Very small helpers for parton distribution functions."""

from __future__ import annotations

from dataclasses import dataclass
from math import pow
from typing import Dict

try:  # pragma: no cover - optional
    import lhapdf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional
    lhapdf = None  # type: ignore

ParticleId = int


class PartonDistribution:
    """Simple interface for different PDF backends."""

    def xfx(self, pid: ParticleId, x: float, q2: float) -> float:
        raise NotImplementedError


class LHAPDFSet(PartonDistribution):
    """Use the LHAPDF library when it is available."""

    def __init__(self, setname: str = "CTEQ6L1", member: int = 0) -> None:
        if lhapdf is None:  # pragma: no cover - depends on install
            raise RuntimeError("LHAPDF is not installed")
        self._pdf = lhapdf.mkPDF(setname, member)

    def xfx(self, pid: ParticleId, x: float, q2: float) -> float:
        if x <= 0.0 or x >= 1.0:
            return 0.0
        return self._pdf.xfxQ2(pid, x, q2) / x


@dataclass(frozen=True)
class BetaParameters:
    """Coefficients for a single beta-like term."""

    amplitude: float
    small_x_power: float
    large_x_power: float

    def evaluate(self, x: float) -> float:
        if x <= 0.0 or x >= 1.0:
            return 0.0
        value = self.amplitude
        value *= pow(x, -self.small_x_power)
        value *= pow(1.0 - x, self.large_x_power)
        return value


class ApproximatePDF(PartonDistribution):
    """Fallback smooth model used in the tests."""

    _SUPPORTED: Dict[ParticleId, str] = {
        -5: "bbar",
        -4: "cbar",
        -3: "sbar",
        -2: "ubar",
        -1: "dbar",
        1: "d",
        2: "u",
        3: "s",
        4: "c",
        5: "b",
    }

    def __init__(self) -> None:
        self._valence: Dict[int, BetaParameters] = {}
        self._valence[1] = BetaParameters(0.65, 0.45, 4.5)
        self._valence[2] = BetaParameters(1.00, 0.40, 3.5)
        self._sea: Dict[int, BetaParameters] = {}
        self._sea[1] = BetaParameters(0.0486, 1.45, 7.0)
        self._sea[2] = BetaParameters(0.0545, 1.43, 6.8)
        self._sea[3] = BetaParameters(0.0396, 1.29, 7.2)
        self._sea[4] = BetaParameters(0.0208, 1.08, 7.5)
        self._sea[5] = BetaParameters(0.0083, 0.98, 8.0)

    def xfx(self, pid: ParticleId, x: float, q2: float) -> float:  # noqa: ARG002
        if pid not in self._SUPPORTED:
            raise ValueError(f"Unsupported parton id {pid}")
        if x <= 0.0 or x >= 1.0:
            return 0.0
        flavour = abs(pid)
        sea = self._sea.get(flavour)
        sea_value = sea.evaluate(x) if sea is not None else 0.0
        if pid > 0:
            valence = self._valence.get(flavour)
            valence_value = valence.evaluate(x) if valence is not None else 0.0
            return valence_value + sea_value
        return sea_value


def make_pdf(setname: str = "CTEQ6L1", member: int = 0) -> PartonDistribution:
    """Return a PDF backend, preferring LHAPDF."""

    if lhapdf is not None:  # pragma: no cover - optional
        return LHAPDFSet(setname=setname, member=member)
    return ApproximatePDF()


__all__ = [
    "ApproximatePDF",
    "BetaParameters",
    "LHAPDFSet",
    "PartonDistribution",
    "make_pdf",
]
