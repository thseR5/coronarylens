"""The deduplicated Owen sampler must equal the plain (evaluate-every-row) implementation exactly."""

import numpy as np

from ml.explain import _orders, owen_values, representative_background


def _owen_plain(f, x, bg, groups, n_perm, seed):
    d, nb = x.size, bg.shape[0]
    orders = _orders(groups, n_perm, seed)
    P = len(orders)
    masks = np.zeros((P, d - 1, d), dtype=bool)
    for p, order in enumerate(orders):
        for j in range(1, d):
            masks[p, j - 1, order[:j]] = True
    Z = np.where(masks.reshape(-1, d)[:, None, :], x[None, None, :], bg[None, :, :]).reshape(-1, d)
    out = f(np.vstack([bg, x[None, :], Z]))
    k = out.shape[1]
    v0, fx = out[:nb].mean(0), out[nb]
    vmid = out[nb + 1:].reshape(P, d - 1, nb, k).mean(2)
    phi = np.zeros((d, k))
    for p, order in enumerate(orders):
        phi[order] += np.diff(np.vstack([v0, vmid[p], fx]), axis=0)
    return phi / P, v0, fx


def test_dedup_matches_plain_implementation():
    rng = np.random.default_rng(0)
    d, nb = 9, 12
    W = rng.normal(size=(d, 3))
    bg = rng.integers(0, 2, size=(nb, d)).astype(float)  # binary-ish background so many rows repeat
    bg[:, :3] = rng.normal(size=(nb, 3))
    x = np.r_[rng.normal(size=3), rng.integers(0, 2, size=d - 3)].astype(float)
    groups = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]

    def f(Z):  # non-additive multi-output function
        return np.tanh(Z @ W) + 0.3 * (Z[:, :1] * Z[:, 3:4])

    a = owen_values(f, x, bg, groups, n_perm=6, seed=3)
    b = _owen_plain(f, x, bg, groups, n_perm=6, seed=3)
    for u, v in zip(a, b):
        np.testing.assert_allclose(u, v, atol=1e-12)
    np.testing.assert_allclose(a[1] + a[0].sum(0), a[2], atol=1e-12)  # additivity


def test_background_is_real_patients():
    rng = np.random.default_rng(1)
    X = np.c_[rng.normal(size=(80, 3)), rng.integers(0, 2, size=(80, 4))]
    bg = representative_background(X, 10, seed=0)
    assert bg.shape == (10, 7)
    assert all(any(np.array_equal(r, x) for x in X) for r in bg)
    assert len({tuple(r) for r in bg}) == 10
