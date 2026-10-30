"""Reproducible FP64 elementary checks, NOT a model-training experiment."""
import numpy as np

rng = np.random.default_rng(20261030)
TOL = 2e-11

def householder(v):
    v = np.asarray(v, dtype=np.float64)
    assert np.linalg.norm(v) > 0
    return np.eye(len(v)) - 2 * np.outer(v, v) / (v @ v)

def relu(z):
    return np.maximum(z, 0)

def phi(Q, z, f=relu):
    return Q.T @ f(Q @ z)

Q = householder(np.array([1., 2.]))
I = np.eye(2)
P = np.array([[0., 1.], [1., 0.]])
assert np.allclose(Q.T @ Q, I, atol=TOL)
assert np.linalg.det(Q) < 0
checks = []

for _ in range(200):
    z = rng.normal(size=2)
    x = rng.normal(size=2)
    # Idempotence, non-expansive, 1-Lipschitz
    p = phi(Q,z)
    np.testing.assert_allclose(phi(Q,p),p,atol=TOL)
    assert np.linalg.norm(p) <= np.linalg.norm(z)+TOL
    assert np.linalg.norm(phi(Q,z)-phi(Q,x)) <= np.linalg.norm(z-x)+TOL
    # Linear activation around itself cancels Q.
    np.testing.assert_allclose(phi(Q,z,lambda a:a), z, atol=TOL)
    # Any coordinate PERMUTATION before Q yields the same ReLU projection.
    np.testing.assert_allclose(phi(P @ Q,z),p,atol=TOL)
    # Positive/negative orthogonal views sum to input (general O(d) views).
    np.testing.assert_allclose(phi(Q,z)+phi(-Q,z),z,atol=TOL)
checks.append('PASS 200 random cases: idempotence, norm decrease, nonexpansive, linear cancellation, permutation symmetry, complement identity')

# Single Householder is equivalent to *coupled* rank-one up/down weight updates.
h,din,dout=6,4,3
Wup=rng.normal(size=(h,din));Wdown=rng.normal(size=(dout,h));
v=rng.normal(size=h);u=v/np.linalg.norm(v);H=householder(v)
for _ in range(200):
    x=rng.normal(size=din)
    lhs=Wdown @ H @ relu(H @ Wup @ x)
    Wup_eff=Wup-2*np.outer(u,u@Wup)
    Wdown_eff=Wdown-2*np.outer(Wdown@u,u)
    rhs=Wdown_eff@relu(Wup_eff@x)
    np.testing.assert_allclose(lhs,rhs,atol=TOL)
checks.append('PASS 200 random cases: Householder nonlinear shared MLP == coupled rank-one up/down weights')

# Gradients of ReLU mirror are orthogonal projectors outside activation boundaries.
for _ in range(100):
    z=rng.normal(size=2)
    if np.min(np.abs(Q@z))<1e-4: continue
    mask=(Q@z>0).astype(float)
    J=Q.T @ np.diag(mask) @ Q
    assert np.linalg.norm(J@J-J)<TOL
    assert np.linalg.norm(J.T-J)<TOL
    for i in range(2):
        e=np.eye(2)[:,i]*1e-6
        obs=(phi(Q,z+e)-phi(Q,z-e))/(2e-6)
        np.testing.assert_allclose(obs,J[:,i],rtol=0,atol=1e-8)
checks.append('PASS finite differences: Jacobian = rotated diagonal binary gate and is projector')

# Nonlinear ReLU mirror differs from standard ReLU for nontrivial Q and input.
z=np.array([1.,-2.]); diff=float(np.linalg.norm(phi(Q,z)-phi(I,z)))
assert diff>0.1
checks.append(f'PASS nontrivial gating: ||Mirror(z)-ReLU(z)||_2={diff:.9f} for z=(1,-2)')

# Repeated same gate has no further effect. Alternating gates can change it.
assert np.linalg.norm(phi(Q,phi(Q,z))-phi(Q,z))<TOL
w=np.array([-1.1,2.3])
a=phi(I,w);b=phi(Q,a)
assert np.linalg.norm(b-a)>1e-3
checks.append(f'PASS repeated identical projection idempotent; switched View alters state by {np.linalg.norm(b-a):.9f}')

# Whole-kernel conjugation does NOT cancel for an anisotropic LINEAR W.
W=np.diag([1.,3.]);
assert np.linalg.norm(Q.T@W@Q-W)>0.1
checks.append(f'PASS nuance: full-kernel Q^T W Q != W for noncommuting linear W (norm {np.linalg.norm(Q.T@W@Q-W):.9f})')

# For a single reflection, full linear-kernel conjugation changes W by rank at most two.
for _ in range(100):
    d=8
    v=rng.normal(size=d);u=v/np.linalg.norm(v);Q8=householder(v)
    W=rng.normal(size=(d,d))
    D=Q8.T@W@Q8-W
    # The exact expression only has two column directions: u and W@u.
    B=np.column_stack([u,W@u])
    approx=B @ np.linalg.lstsq(B,D,rcond=None)[0]
    assert np.linalg.norm(D-approx)<1e-10
checks.append('PASS 100 random cases: linear whole-kernel Householder conjugation delta column-rank <= 2')

# The leading view-dependent variation is first/second/third order for ReLU/SiLU/tanh.
def silu(x): return x/(1+np.exp(-x))
def tanh(x): return np.tanh(x)
z=np.array([0.71,-1.04],dtype=float)
for activation,power in [(relu,1),(silu,2),(tanh,3)]:
    e=0.025
    f=lambda scale: np.linalg.norm(phi(Q,scale*z,activation)-phi(I,scale*z,activation))
    a,b=f(e),f(e/2)
    assert a>1e-10 and b>1e-12
    slope=np.log(a/b)/np.log(2)
    assert abs(slope-power)<0.02,(activation.__name__,slope)
    checks.append(f'PASS near-zero amplitude exponent for {activation.__name__}: {slope:.6f} (expected {power})')

# Pure coefficient counts: K full identical-width MLPs vs one core + K Householders.
P0=h*din+h+dout*h+dout
K=4
PM=P0+K*h
PD=K*P0
assert PM<PD
checks.append(f'PASS conditional coefficient count: h={h} din={din} dout={dout} K={K}, shared={PM}, independent={PD}; excludes extra routing and actual bytes')

print('NumPy version:',np.__version__)
print('Seed: 20261030, FP64, CPU')
for line in checks: print(line)
print('SUCCESS',len(checks),'check groups; no model training conducted')
