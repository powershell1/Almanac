import numpy as np
import pytest

from gdr4.model.kepler import orbit_xy, solve_kepler


def test_kepler_residual_below_1e12():
    m = np.linspace(-20 * np.pi, 20 * np.pi, 4001)
    for e in np.linspace(0.0, 0.95, 39):
        ecc_anom = solve_kepler(m, e)
        resid = ecc_anom - e * np.sin(ecc_anom) - m
        assert np.max(np.abs(resid)) < 1e-12, e


def test_kepler_edge_cases():
    assert solve_kepler(0.0, 0.9) == 0.0
    np.testing.assert_allclose(solve_kepler(np.pi, 0.7), np.pi, atol=1e-14)
    m = np.array([1e-9, -1e-9, 2 * np.pi - 1e-9])
    resid = solve_kepler(m, 0.95) - 0.95 * np.sin(solve_kepler(m, 0.95)) - m
    assert np.max(np.abs(resid)) < 1e-12


def test_kepler_broadcasting():
    m = np.linspace(0, 6, 7)
    e = np.array([[0.1], [0.5], [0.9]])
    assert solve_kepler(m, e).shape == (3, 7)


def test_kepler_rejects_bad_ecc():
    with pytest.raises(ValueError):
        solve_kepler(1.0, 1.0)


def test_orbit_xy_circular():
    t = np.linspace(-500, 500, 11)
    x, y = orbit_xy(t, 100.0, 0.0, 0.0)
    phase = 2 * np.pi * t / 100.0
    np.testing.assert_allclose(x, np.cos(phase), atol=1e-13)
    np.testing.assert_allclose(y, np.sin(phase), atol=1e-13)


def test_matches_gaiamock_c_solver(gaiamock, gaiamock_c):
    m = np.linspace(-30.0, 30.0, 2001)
    for e in (0.0, 0.3, 0.7, 0.95):
        ref = gaiamock.solve_kepler_eqn_on_array(m, e, gaiamock_c, xtol=1e-14)
        # gaiamock may return E modulo 2 pi; compare on the circle.
        diff = np.angle(np.exp(1j * (solve_kepler(m, e) - ref)))
        assert np.max(np.abs(diff)) < 1e-9, e
