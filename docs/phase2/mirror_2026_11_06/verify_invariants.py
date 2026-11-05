"""Strictly algebraic Mirror invariants; not a trained-model benchmark."""
import numpy as np

rng = np.random.default_rng(20261106)

def softmax(x):
    y = np.exp(x - np.max(x))
    return y / np.sum(y)

max_kv = 0.0
for _ in range(256):
    d, dv, n = 8, 5, 13
    q = rng.normal(size=d)
    keys = rng.normal(size=(n, d))
    values = rng.normal(size=(n, dv))
    a = rng.normal(size=(d, d))
    b = rng.normal(size=(dv, dv))
    direct = softmax(q @ (keys @ a).T / np.sqrt(d)) @ (values @ b)
    lazy = (softmax((q @ a.T) @ keys.T / np.sqrt(d)) @ values) @ b
    max_kv = max(max_kv, np.max(np.abs(direct - lazy)))
assert max_kv < 5e-12

max_rank_error = 0
max_orth = 0.0
for r in range(1, 5):
    for _ in range(64):
        d = 32
        q = np.eye(d)
        for _ in range(r):
            u = rng.normal(size=d)
            u /= np.linalg.norm(u)
            h = np.eye(d) - 2 * np.outer(u, u)
            q = h @ q
        rank = np.linalg.matrix_rank(q - np.eye(d), tol=1e-9)
        max_rank_error = max(max_rank_error, rank-r)
        max_orth = max(max_orth, np.max(np.abs(q.T@q-np.eye(d))))
assert max_rank_error <= 0
assert max_orth < 1e-12

# A genuinely different input rotation need not be representable by an
# orthogonal left transform of a fixed anisotropic input projection.
a = np.diag([2.0, 1.0])
r = np.array([[0.0, 1.0], [1.0, 0.0]])
q_required = a @ r @ np.linalg.inv(a)
assert np.max(np.abs(q_required.T@q_required-np.eye(2))) > 0.1

# Restricted coupled-rank1 map includes the single Householder exactly.
for _ in range(128):
    u = rng.normal(size=16)
    u /= np.linalg.norm(u)
    h = np.eye(16) - 2*np.outer(u, u)
    s = np.eye(16) + (-2)*np.outer(u, u)
    assert np.array_equal(h, s)

print('PASS 256 FP64 canonical-KV right-transform identities, max abs discrepancy %.3e' % max_kv)
print('PASS 256 FP64 Householder rank<=r (r=1..4, d=32), max orth error %.3e' % max_orth)
print('PASS anisotropic input representation nonorthogonal-required example')
print('PASS 128 single-Householder / coupled-rank1 exact inclusions')
print('NO TRAINING, REAL-MODEL QUALITY, GPU SPEED OR NATURAL-LANGUAGE CLAIM')
