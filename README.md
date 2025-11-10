# PH121 Numerical Integration Helpers

This repository contains small, dependency-free Python utilities that support the
PH121 assignment on numerical integration, chaotic dynamical systems, and Monte Carlo
techniques.

## Modules

### `ph121.integrators`

Utilities for solving systems of ordinary differential equations.

* `forward_euler(derivative, initial_state, *, t0, t_end, step)` – explicit first order method.
* `runge_kutta4(derivative, initial_state, *, t0, t_end, step)` – classic fourth order Runge–Kutta.
* `velocity_verlet(acceleration, initial_position, initial_velocity, *, t0, t_end, step)` – symplectic scheme for Newtonian systems.

All functions return an `IntegrationResult` dataclass containing the sampled times and states. The
states are stored as tuples and can be converted to lists via `result.as_lists()`.

### `ph121.mercury`

Numerical helpers for simulating Mercury's orbit with the leading-order general relativity
correction.

* `integrate_mercury_orbit(duration, step, *, relativistic)` – evolve the orbit and return the
  radial, angular, and energy time series.
* `estimate_precession(duration_years=100.0, step_days=0.25)` – compare relativistic and
  Newtonian integrations to estimate the perihelion advance in arcseconds per century.
* `plot_conserved_quantities(result)` – visualise the fractional drift of the energy and angular
  momentum for a simulation.

### `ph121.monte_carlo`

Monte Carlo integration utilities covering uniform sampling, general importance sampling, and a
Breit–Wigner focused helper.

* `monte_carlo_integrate(func, bounds, samples, *, rng=None)` – estimate an integral over a hyper-rectangle using uniform random points.
* `importance_sampling(func, sampler, weight, samples, *, rng=None)` – evaluate arbitrary integrals using user-provided proposal distributions.
* `breit_wigner_sampler(mass, width, *, rho_bounds=None)` / `breit_wigner_weight(mass, width, *, rho_bounds=None)` – convenience helpers implementing the change of variables discussed in the assignment handout.

### `ph121.high_energy`

Monte Carlo evaluations of the high-energy scattering problems from the assignment.

* `total_cross_section_electron_muon(energy, samples, *, rng=None)` – integrates the tree-level ``e^+e^- \to \gamma \to \mu^+\mu^-`` cross section over angles and compares with the analytic result.
* `drell_yan_total_cross_section(energy, samples, *, q_min=60.0, rng=None, pdf=None)` – computes the proton–proton ``\gamma/Z`` Drell–Yan cross section using importance sampling and either LHAPDF or the built-in analytic PDF model.
* `drell_yan_differential_cross_section(energy, samples, bins, *, q_min=60.0, rng=None, pdf=None)` – returns the total cross section alongside a ``dσ/dτ`` histogram extracted from the Monte Carlo samples.

The accompanying :mod:`ph121.pdfs` module exposes ``make_pdf()`` which prefers LHAPDF when installed and otherwise uses a light-weight analytic parametrisation tuned to reproduce the benchmark LHC cross sections.

## Quick example

```python
from math import sin
from ph121.integrators import runge_kutta4

# Simple harmonic oscillator: y'' + y = 0 rewritten as a first-order system.
def harmonic_derivative(t, state):
    y, v = state
    return (v, -y)

solution = runge_kutta4(harmonic_derivative, (1.0, 0.0), t0=0.0, t_end=10.0, step=0.1)
print(solution.states[-1])
```

```python
from ph121.monte_carlo import monte_carlo_integrate

# Estimate \int_0^1 sin(x) dx.
def integrand(point):
    (x,) = point
    return sin(x)

result = monte_carlo_integrate(integrand, bounds=[(0.0, 1.0)], samples=10000)
print(result.value, result.error)
```
