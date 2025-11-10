"""Simple Mercury perihelion tools without extra dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from typing import List, Sequence, Tuple

G = 6.67430e-11
C = 299_792_458.0
M_SUN = 1.98847e30
M_MERCURY = 3.3011e23
MERCURY_SEMIMAJOR_AXIS = 5.7909175e10
MERCURY_ECCENTRICITY = 0.205630
MERCURY_ORBITAL_PERIOD = 87.9691 * 86_400.0
SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY


@dataclass(frozen=True)
class MercuryOrbitResult:
    """Store a time series for the orbit integration."""

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
        return sum(self.angular_momenta) / float(len(self.angular_momenta))

    def cartesian_positions(self) -> Tuple[Tuple[float, float], ...]:
        points: List[Tuple[float, float]] = []
        for radius, angle in zip(self.radii, self.angles):
            points.append((radius * cos(angle), radius * sin(angle)))
        return tuple(points)

    def cartesian_velocities(self) -> Tuple[Tuple[float, float], ...]:
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
    return M_SUN * M_MERCURY / (M_SUN + M_MERCURY)


def conserved_angular_momentum() -> float:
    reduced_mass = reduced_mass_sun_mercury()
    total_mass = M_SUN + M_MERCURY
    specific = sqrt(G * total_mass * MERCURY_SEMIMAJOR_AXIS * (1.0 - MERCURY_ECCENTRICITY**2))
    return reduced_mass * specific


def _effective_potential(radius: float, reduced_mass: float, angular_momentum: float, relativistic: bool) -> float:
    base = -G * M_SUN * M_MERCURY / radius
    centrifugal = angular_momentum**2 / (2.0 * reduced_mass * radius * radius)
    if not relativistic:
        correction = 0.0
    else:
        correction = (
            -G * (M_SUN + M_MERCURY) * angular_momentum**2 / (C * C * reduced_mass * radius**3)
        )
    return base + centrifugal + correction


def _radial_force(radius: float, reduced_mass: float, angular_momentum: float, relativistic: bool) -> float:
    l_term = angular_momentum**2 / (reduced_mass * radius**3)
    gravity = G * M_SUN * M_MERCURY / (radius * radius)
    if not relativistic:
        correction = 0.0
    else:
        beta = G * (M_SUN + M_MERCURY) / (C * C)
        correction = 3.0 * beta * angular_momentum**2 / (reduced_mass * radius**4)
    return l_term - gravity - correction


def _theta_rate(radius: float, reduced_mass: float, angular_momentum: float) -> float:
    return angular_momentum / (reduced_mass * radius * radius)


def _initial_conditions() -> Tuple[float, float, float, float]:
    reduced_mass = reduced_mass_sun_mercury()
    r0 = MERCURY_SEMIMAJOR_AXIS * (1.0 - MERCURY_ECCENTRICITY)
    theta0 = 0.0
    p_r0 = 0.0
    L = conserved_angular_momentum()
    return r0, theta0, p_r0, L


def integrate_mercury_orbit(*, duration: float, step: float, relativistic: bool) -> MercuryOrbitResult:
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
    while t <= duration + 1.0e-12:
        times.append(t)
        radii.append(r)
        angles.append(theta)
        radial_momenta.append(p_r)
        energies.append(p_r * p_r / (2.0 * reduced_mass) + _effective_potential(r, reduced_mass, L, relativistic))
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
        if next_time >= duration:
            break
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


def _estimate_angular_momentum(times: Sequence[float], radii: Sequence[float], angles: Sequence[float]) -> Tuple[float, ...]:
    reduced_mass = reduced_mass_sun_mercury()
    values: List[float] = []
    last_index = len(times) - 1
    for index, radius in enumerate(radii):
        if index == 0:
            dt = times[1] - times[0]
            dtheta = angles[1] - angles[0]
        elif index == last_index:
            dt = times[-1] - times[-2]
            dtheta = angles[-1] - angles[-2]
        else:
            dt = times[index + 1] - times[index - 1]
            dtheta = angles[index + 1] - angles[index - 1]
        theta_rate = dtheta / dt
        values.append(reduced_mass * radius * radius * theta_rate)
    return tuple(values)


def perihelion_angles(result: MercuryOrbitResult) -> Tuple[float, ...]:
    if len(result.radii) < 3:
        return ()
    angles: List[float] = []
    for index in range(1, len(result.radii) - 1):
        r_prev = result.radii[index - 1]
        r_curr = result.radii[index]
        r_next = result.radii[index + 1]
        if not (r_curr <= r_prev and r_curr <= r_next):
            continue
        denom = r_prev - 2.0 * r_curr + r_next
        if denom == 0.0:
            offset = 0.0
        else:
            offset = (r_prev - r_next) / (2.0 * denom)
        theta_interp = result.angles[index] + offset * (result.angles[index + 1] - result.angles[index])
        angles.append(theta_interp)
    return tuple(angles)


@dataclass(frozen=True)
class PrecessionEstimate:
    """Store Newtonian and relativistic runs."""

    relativistic: MercuryOrbitResult
    newtonian: MercuryOrbitResult
    relativistic_perihelia: Tuple[float, ...]
    newtonian_perihelia: Tuple[float, ...]
    arcseconds_per_century: float


def estimate_precession(*, duration_years: float = 100.0, step_days: float = 0.25) -> PrecessionEstimate:
    duration = duration_years * SECONDS_PER_YEAR
    step = step_days * SECONDS_PER_DAY
    relativistic = integrate_mercury_orbit(duration=duration, step=step, relativistic=True)
    newtonian = integrate_mercury_orbit(duration=duration, step=step, relativistic=False)
    rel_perihelia = perihelion_angles(relativistic)
    newt_perihelia = perihelion_angles(newtonian)
    if len(rel_perihelia) < 2 or len(newt_perihelia) < 2:
        raise RuntimeError("integration too short")
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


def plot_conserved_quantities(result: MercuryOrbitResult, *, show: bool = True) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional
        raise RuntimeError("matplotlib is required") from exc
    energy0 = result.energies[0]
    angular0 = result.angular_momenta[0]
    delta_energy: List[float] = []
    delta_angular: List[float] = []
    for energy in result.energies:
        delta_energy.append((energy - energy0) / abs(energy0))
    for angular in result.angular_momenta:
        delta_angular.append((angular - angular0) / abs(angular0))
    times = [time / SECONDS_PER_YEAR for time in result.times]
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(times, delta_energy)
    ax1.set_ylabel(r"$\Delta H / H_0$")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax2.plot(times, delta_angular)
    ax2.set_ylabel(r"$\Delta L / L_0$")
    ax2.set_xlabel("Time [years]")
    ax2.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle(("Relativistic" if result.relativistic else "Newtonian") + " Mercury Orbit Conservation")
    if show:  # pragma: no cover - plotting
        plt.show()


__all__ = [
    "MERCURY_ECCENTRICITY",
    "MERCURY_ORBITAL_PERIOD",
    "MERCURY_SEMIMAJOR_AXIS",
    "MercuryOrbitResult",
    "PrecessionEstimate",
    "conserved_angular_momentum",
    "estimate_precession",
    "integrate_mercury_orbit",
    "perihelion_angles",
    "plot_conserved_quantities",
    "reduced_mass_sun_mercury",
]
