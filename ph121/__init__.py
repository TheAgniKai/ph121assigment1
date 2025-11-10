"""Utility modules for PH121 numerical integration assignment."""

from .integrators import forward_euler, runge_kutta4, velocity_verlet
from .chaos import (
    BoxCountingEstimate,
    InformationDimensionEstimate,
    LyapunovEstimate,
    LyapunovSpectrumEstimate,
    estimate_lorenz_box_counting_dimension,
    estimate_lorenz_information_dimension,
    estimate_lorenz_maximal_lyapunov,
    estimate_lorenz_lyapunov_spectrum,
    format_estimates,
    lorenz63_derivative,
    lorenz63_jacobian,
)
from .mercury import (
    PrecessionEstimate,
    estimate_precession,
    integrate_mercury_orbit,
    perihelion_angles,
    plot_conserved_quantities,
)
from .monte_carlo import (
    monte_carlo_integrate,
    importance_sampling,
    breit_wigner_sampler,
    breit_wigner_weight,
)

__all__ = [
    "forward_euler",
    "runge_kutta4",
    "velocity_verlet",
    "BoxCountingEstimate",
    "InformationDimensionEstimate",
    "LyapunovEstimate",
    "LyapunovSpectrumEstimate",
    "estimate_lorenz_box_counting_dimension",
    "estimate_lorenz_information_dimension",
    "estimate_lorenz_maximal_lyapunov",
    "estimate_lorenz_lyapunov_spectrum",
    "format_estimates",
    "lorenz63_derivative",
    "lorenz63_jacobian",
    "PrecessionEstimate",
    "estimate_precession",
    "integrate_mercury_orbit",
    "perihelion_angles",
    "plot_conserved_quantities",
    "monte_carlo_integrate",
    "importance_sampling",
    "breit_wigner_sampler",
    "breit_wigner_weight",
]
