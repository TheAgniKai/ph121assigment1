"""Mercury perihelion precession simulation utilities.

The helpers in this module implement the relativistic correction described in
Problem 6.2 of the PH121 assignment.  The dynamics are formulated in polar
coordinates, using the conservation of angular momentum to reduce the problem
to a single radial degree of freedom with the effective potential

.. math::

    V_\text{eff}(r) = -\frac{GM_\odot m}{r} + \frac{L^2}{2\mu r^2}
    - \frac{G(M_\odot + m)L^2}{c^2\mu r^3},

where :math:`\mu` is the reduced mass and :math:`L = \mu r^2\dot{\theta}` is the
conserved angular momentum.  The resulting one-dimensional Hamiltonian is
separable, allowing for an accurate second-order symplectic integrator that
keeps the energy and angular momentum nearly constant over long integrations.

Two public helpers are provided:

``integrate_mercury_orbit``
    Perform a single integration, returning the time series for the radial
    coordinate, polar angle, canonical radial momentum, energy, and angular
    momentum.

``estimate_precession``
    Integrate both the relativistic and purely Newtonian problems and measure
    the cumulative perihelion shift over the requested span of time.

The functions are dependency-free; plotting the conserved quantities merely
requires the optional :mod:`matplotlib` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from typing import List, Sequence, Tuple

# Physical constants describing the Sun–Mercury system.
G = 6.67430e-11  # gravitational constant [m^3 s^-2 kg^-1]
C = 299_792_458.0  # speed of light [m s^-1]
M_SUN = 1.98847e30  # mass of the Sun [kg]
M_MERCURY = 3.3011e23  # mass of Mercury [kg]

# Orbital parameters for Mercury.
MERCURY_SEMIMAJOR_AXIS = 5.7909175e10  # metres
MERCURY_ECCENTRICITY = 0.205630
MERCURY_ORBITAL_PERIOD = 87.9691 * 86_400.0  # seconds

SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY


@dataclass(frozen=True)
class MercuryOrbitResult:
    """Time series describing a simulated Mercury orbit."""

    times: Tuple[float, ...]
    radii: Tuple[float, ...]
    angles: Tuple[float, ...]
    radial_momenta: Tuple[float, ...]
    energies: Tuple[float, ...]
    angular_momenta: Tuple[float, ...]
    relativistic: bool
    step: float

    @property
    def angular_momentum(self) -> float:
        """Return the mean angular momentum from the simulation."""

        return sum(self.angular_momenta) / float(len(self.angular_momenta))

    def cartesian_positions(self) -> Tuple[Tuple[float, float], ...]:
        """Return the Cartesian positions corresponding to the orbit."""

        return tuple(
            (radius * cos(angle), radius * sin(angle))
            for radius, angle in zip(self.radii, self.angles)
        )

    def cartesian_velocities(self) -> Tuple[Tuple[float, float], ...]:
        """Return the Cartesian velocities reconstructed from the state."""

        reduced_mass = reduced_mass_sun_mercury()
        angular_momentum = self.angular_momentum
        velocities: List[Tuple[float, float]] = []
        for radius, angle, p_r in zip(self.radii, self.angles, self.radial_momenta):
            radial_velocity = p_r / reduced_mass
            angular_velocity = angular_momentum / (reduced_mass * radius * radius)
            vx = radial_velocity * cos(angle) - radius * angular_velocity * sin(angle)
            vy = radial_velocity * sin(angle) + radius * angular_velocity * cos(angle)
            velocities.append((vx, vy))
        return tuple(velocities)


def reduced_mass_sun_mercury() -> float:
    """Return the reduced mass of the Sun–Mercury system."""

    return M_SUN * M_MERCURY / (M_SUN + M_MERCURY)


def conserved_angular_momentum() -> float:
    """Return the angular momentum for Mercury at perihelion."""

    reduced_mass = reduced_mass_sun_mercury()
    total_mass = M_SUN + M_MERCURY
    specific_angular_momentum = sqrt(
        G * total_mass * MERCURY_SEMIMAJOR_AXIS * (1.0 - MERCURY_ECCENTRICITY**2)
    )
    return reduced_mass * specific_angular_momentum


def _effective_potential(
    radius: float,
    reduced_mass: float,
    angular_momentum: float,
    relativistic: bool,
) -> float:
    """Compute the effective potential at *radius*."""

    base = -G * M_SUN * M_MERCURY / radius
    centrifugal = angular_momentum**2 / (2.0 * reduced_mass * radius * radius)
    if not relativistic:
        correction = 0.0
    else:
        correction = (
            -G
            * (M_SUN + M_MERCURY)
            * angular_momentum**2
            / (C * C * reduced_mass * radius**3)
        )
    return base + centrifugal + correction


def _radial_force(
    radius: float,
    reduced_mass: float,
    angular_momentum: float,
    relativistic: bool,
) -> float:
    """Return the radial force :math:`\dot{p}_r` for the effective potential."""

    l_term = angular_momentum**2 / (reduced_mass * radius**3)
    gravity = G * M_SUN * M_MERCURY / (radius * radius)
    if not relativistic:
        correction = 0.0
    else:
        beta = G * (M_SUN + M_MERCURY) / (C * C)
        correction = 3.0 * beta * angular_momentum**2 / (reduced_mass * radius**4)
    return l_term - gravity - correction


def _theta_rate(radius: float, reduced_mass: float, angular_momentum: float) -> float:
    """Return the angular velocity using angular momentum conservation."""

    return angular_momentum / (reduced_mass * radius * radius)


def _initial_conditions() -> Tuple[float, float, float, float]:
    """Return ``(r0, theta0, p_r0, L)`` at Mercury's perihelion."""

    reduced_mass = reduced_mass_sun_mercury()
    r0 = MERCURY_SEMIMAJOR_AXIS * (1.0 - MERCURY_ECCENTRICITY)
    theta0 = 0.0
    p_r0 = 0.0  # radial velocity vanishes at perihelion
    L = conserved_angular_momentum()
    return r0, theta0, p_r0, L


def integrate_mercury_orbit(
    *,
    duration: float,
    step: float,
    relativistic: bool,
) -> MercuryOrbitResult:
    """Integrate Mercury's orbit for *duration* seconds.

    Parameters
    ----------
    duration:
        Total integration time in seconds.
    step:
        Integration step size in seconds.
    relativistic:
        If ``True`` include the general relativity correction.
    """

    if step <= 0.0:
        raise ValueError("step must be positive")
    if duration <= 0.0:
        raise ValueError("duration must be positive")

    reduced_mass = reduced_mass_sun_mercury()
    r, theta, p_r, L = _initial_conditions()
    times: List[float] = []
    radii: List[float] = []
    angles: List[float] = []
    radial_momenta: List[float] = []
    energies: List[float] = []

    t = 0.0
    while t <= duration + 1e-9:
        times.append(t)
        radii.append(r)
        angles.append(theta)
        radial_momenta.append(p_r)
        energies.append(
            p_r * p_r / (2.0 * reduced_mass)
            + _effective_potential(r, reduced_mass, L, relativistic)
        )
        angular_velocity = _theta_rate(r, reduced_mass, L)

        next_time = min(duration, t + step)
        h = next_time - t
        if h <= 0.0:
            break

        force = _radial_force(r, reduced_mass, L, relativistic)
        p_half = p_r + 0.5 * h * force
        r_next = r + h * (p_half / reduced_mass)
        next_force = _radial_force(r_next, reduced_mass, L, relativistic)
        p_next = p_half + 0.5 * h * next_force

        theta_rate_current = angular_velocity
        theta_rate_next = _theta_rate(r_next, reduced_mass, L)
        theta += 0.5 * h * (theta_rate_current + theta_rate_next)

        r = r_next
        p_r = p_next
        t = next_time

    angular_momenta = _estimate_angular_momentum(times, radii, angles)

    return MercuryOrbitResult(
        times=tuple(times),
        radii=tuple(radii),
        angles=tuple(angles),
        radial_momenta=tuple(radial_momenta),
        energies=tuple(energies),
        angular_momenta=angular_momenta,
        relativistic=relativistic,
        step=step,
    )


def _estimate_angular_momentum(
    times: Sequence[float],
    radii: Sequence[float],
    angles: Sequence[float],
) -> Tuple[float, ...]:
    """Estimate the angular momentum using finite differences."""

    reduced_mass = reduced_mass_sun_mercury()
    values: List[float] = []
    last_index = len(times) - 1
    for idx, radius in enumerate(radii):
        if idx == 0:
            dt = times[1] - times[0]
            dtheta = angles[1] - angles[0]
        elif idx == last_index:
            dt = times[-1] - times[-2]
            dtheta = angles[-1] - angles[-2]
        else:
            dt = times[idx + 1] - times[idx - 1]
            dtheta = angles[idx + 1] - angles[idx - 1]
        theta_rate = dtheta / dt
        values.append(reduced_mass * radius * radius * theta_rate)
    return tuple(values)


def perihelion_angles(result: MercuryOrbitResult) -> Tuple[float, ...]:
    """Locate perihelion passages and return their polar angles."""

    if len(result.radii) < 3:
        return ()

    angles: List[float] = []
    dt = result.step
    for idx in range(1, len(result.radii) - 1):
        r_prev = result.radii[idx - 1]
        r_curr = result.radii[idx]
        r_next = result.radii[idx + 1]
        if not (r_curr <= r_prev and r_curr <= r_next):
            continue
        denom = r_prev - 2.0 * r_curr + r_next
        if denom == 0.0:
            offset = 0.0
        else:
            offset = (r_prev - r_next) / (2.0 * denom)
        theta_interp = result.angles[idx] + offset * (
            result.angles[idx + 1] - result.angles[idx]
        )
        angles.append(theta_interp)

    return tuple(angles)


@dataclass(frozen=True)
class PrecessionEstimate:
    """Summary of the Newtonian versus relativistic Mercury orbits."""

    relativistic: MercuryOrbitResult
    newtonian: MercuryOrbitResult
    relativistic_perihelia: Tuple[float, ...]
    newtonian_perihelia: Tuple[float, ...]
    arcseconds_per_century: float


def estimate_precession(
    *,
    duration_years: float = 100.0,
    step_days: float = 0.25,
) -> PrecessionEstimate:
    """Estimate the perihelion advance over the requested time span.

    Parameters
    ----------
    duration_years:
        Total simulated time in Julian years.
    step_days:
        Integration step size in days.
    """

    duration = duration_years * SECONDS_PER_YEAR
    step = step_days * SECONDS_PER_DAY

    relativistic = integrate_mercury_orbit(
        duration=duration,
        step=step,
        relativistic=True,
    )
    newtonian = integrate_mercury_orbit(
        duration=duration,
        step=step,
        relativistic=False,
    )

    rel_perihelia = perihelion_angles(relativistic)
    newt_perihelia = perihelion_angles(newtonian)

    if len(rel_perihelia) < 2 or len(newt_perihelia) < 2:
        raise RuntimeError("Integration span too short to locate multiple perihelia")

    count = min(len(rel_perihelia), len(newt_perihelia))
    rel_span = rel_perihelia[count - 1] - rel_perihelia[0]
    newt_span = newt_perihelia[count - 1] - newt_perihelia[0]
    delta = rel_span - newt_span
    arcseconds = delta * (180.0 / pi) * 3600.0
    arcseconds_per_century = arcseconds * (100.0 / duration_years)

    return PrecessionEstimate(
        relativistic=relativistic,
        newtonian=newtonian,
        relativistic_perihelia=rel_perihelia[:count],
        newtonian_perihelia=newt_perihelia[:count],
        arcseconds_per_century=arcseconds_per_century,
    )


def plot_conserved_quantities(
    result: MercuryOrbitResult,
    *,
    show: bool = True,
):
    """Plot the fractional drift of the energy and angular momentum."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting") from exc

    energy0 = result.energies[0]
    angular0 = result.angular_momenta[0]
    delta_energy = [
        (energy - energy0) / abs(energy0) for energy in result.energies
    ]
    delta_angular = [
        (angular - angular0) / abs(angular0)
        for angular in result.angular_momenta
    ]

    times = [t / SECONDS_PER_YEAR for t in result.times]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(times, delta_energy)
    ax1.set_ylabel(r"$\Delta H / H_0$")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(times, delta_angular)
    ax2.set_ylabel(r"$\Delta L / L_0$")
    ax2.set_xlabel("Time [years]")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Relativistic" if result.relativistic else "Newtonian"
        + " Mercury Orbit Conservation"
    )

    if show:  # pragma: no cover - simple plotting helper
        plt.show()
    return fig


__all__ = [
    "MercuryOrbitResult",
    "PrecessionEstimate",
    "conserved_angular_momentum",
    "estimate_precession",
    "integrate_mercury_orbit",
    "perihelion_angles",
    "plot_conserved_quantities",
    "reduced_mass_sun_mercury",
]

