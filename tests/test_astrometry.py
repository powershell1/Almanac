import ctypes

import numpy as np

from gdr4.model.astrometry import (
    DAYS_PER_YEAR,
    campbell_from_thiele_innes,
    design_matrix,
    mean_anomaly_from_t_peri,
    photocentre_semimajor_axis,
    predict_w,
    t_peri_from_mean_anomaly,
    thiele_innes_from_campbell,
)


def _fake_scan(rng, n=90):
    t_d = np.sort(rng.uniform(-990, 990, n))
    psi = rng.uniform(-np.pi, np.pi, n)
    plx = rng.uniform(-1, 1, n)
    return t_d, psi, plx


def _gaiamock_12par(c_funcs, t_d, psi, plx, theta12, campbell=False):
    """Noise-free gaiamock prediction via its C residual function (obs = 0, err = 1)."""
    t_yr = (t_d / DAYS_PER_YEAR).astype(np.double)
    n = len(t_yr)
    out = np.zeros(n)
    zeros, ones = np.zeros(n), np.ones(n)
    psi, plx, theta12 = (np.ascontiguousarray(a, dtype=np.double) for a in (psi, plx, theta12))
    fn = (
        c_funcs.get_residual_array_12par_solution_campbell
        if campbell
        else c_funcs.get_residual_array_12par_solution
    )
    fn(
        ctypes.c_int(n),
        *(ctypes.c_void_p(a.ctypes.data) for a in (t_yr, psi, plx, zeros, ones)),
        ctypes.c_void_p(theta12.ctypes.data),
        ctypes.c_void_p(out.ctypes.data),
    )
    return out


def test_phase_conventions_roundtrip():
    period = np.array([31.0, 400.0, 1999.0])
    m0 = np.array([0.1, 3.0, 6.2])
    tp = t_peri_from_mean_anomaly(m0, period)
    np.testing.assert_allclose(mean_anomaly_from_t_peri(tp, period), m0, atol=1e-12)


def test_design_matrix_reproduces_explicit_formula(rng):
    t_d, psi, plx = _fake_scan(rng)
    theta = rng.normal(size=9)
    period, ecc, tp = 321.0, 0.4, 55.0
    w = predict_w(t_d, psi, plx, period, ecc, tp, theta)
    from gdr4.model.kepler import orbit_xy

    x, y = orbit_xy(t_d, period, ecc, tp)
    dra, ddec, plx_mas, pmra, pmdec, a, b, f, g = theta
    t_yr = t_d / DAYS_PER_YEAR
    expected = (
        (dra + pmra * t_yr) * np.sin(psi)
        + (ddec + pmdec * t_yr) * np.cos(psi)
        + plx_mas * plx
        + (b * x + g * y) * np.sin(psi)
        + (a * x + f * y) * np.cos(psi)
    )
    np.testing.assert_allclose(w, expected, rtol=0, atol=1e-12)


def test_design_matrix_batched_shapes(rng):
    t_d, psi, plx = _fake_scan(rng, n=50)
    s = 7
    m = design_matrix(
        t_d,
        psi,
        plx,
        rng.uniform(30, 2000, (s, 1)),
        rng.uniform(0, 0.9, (s, 1)),
        rng.uniform(-100, 100, (s, 1)),
    )
    assert m.shape == (s, 50, 9)


def test_reproduces_gaiamock_abscissae(rng, gaiamock_c):
    """Same inputs -> same noise-free abscissae as gaiamock's C forward model."""
    t_d, psi, plx = _fake_scan(rng, n=120)
    for _ in range(20):
        period = rng.uniform(30, 2000)
        ecc = rng.uniform(0, 0.95)
        m0 = rng.uniform(0, 2 * np.pi)
        lin = rng.normal(0, 3, 9)
        # gaiamock theta = (ra_off, dec_off, plx, pmra, pmdec, P, e, phi_p, A, B, F, G), phi_p = -M0
        theta12 = np.r_[lin[:5], period, ecc, -m0, lin[5:]]
        ref = _gaiamock_12par(gaiamock_c, t_d, psi, plx, theta12)
        ours = predict_w(t_d, psi, plx, period, ecc, t_peri_from_mean_anomaly(m0, period), lin)
        # gaiamock's C solver stops at xtol = 1e-10
        np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-8)


def test_campbell_conversion_matches_gaiamock_c(rng, gaiamock_c):
    t_d, psi, plx = _fake_scan(rng, n=80)
    for _ in range(10):
        period, ecc, m0 = rng.uniform(30, 2000), rng.uniform(0, 0.9), rng.uniform(0, 2 * np.pi)
        argp, node, a0, inc = rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi), 2.0, 1.0
        five = rng.normal(0, 3, 5)
        theta_c = np.r_[five, period, ecc, -m0, argp, node, a0, inc]
        ref = _gaiamock_12par(gaiamock_c, t_d, psi, plx, theta_c, campbell=True)
        ti = thiele_innes_from_campbell(a0, node, argp, inc)
        ours = predict_w(
            t_d, psi, plx, period, ecc, t_peri_from_mean_anomaly(m0, period), np.r_[five, ti]
        )
        np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-8)


def test_campbell_roundtrip(rng):
    n = 2000
    a0 = rng.uniform(0.1, 10, n)
    node = rng.uniform(0, np.pi, n)
    argp = rng.uniform(0, 2 * np.pi, n)
    inc = np.arccos(rng.uniform(-1, 1, n))
    a, b, f, g = thiele_innes_from_campbell(a0, node, argp, inc)
    a0_r, node_r, argp_r, inc_r = campbell_from_thiele_innes(a, b, f, g)
    np.testing.assert_allclose(a0_r, a0, rtol=1e-9)
    # Reconstructed elements must give back the same Thiele-Innes constants.
    np.testing.assert_allclose(
        np.array(thiele_innes_from_campbell(a0_r, node_r, argp_r, inc_r)),
        np.array([a, b, f, g]),
        atol=1e-8,
    )


def test_campbell_matches_gaiamock_python(rng, gaiamock):
    a, b, f, g = rng.normal(0, 1, (4, 500))
    ours = campbell_from_thiele_innes(a, b, f, g)
    ref = gaiamock.get_Campbell_elements(a, b, f, g)  # (a0, Omega, w, inc)
    for x, y in zip(ours, ref, strict=True):
        np.testing.assert_allclose(x, y, atol=1e-12)


def test_semimajor_axis_scalar():
    a, b, f, g = thiele_innes_from_campbell(3.0, 0.3, 1.2, 0.7)
    np.testing.assert_allclose(photocentre_semimajor_axis(a, b, f, g), 3.0, rtol=1e-12)
