import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

here = Path(__file__).resolve().parent
plots = here / "plots"
plots.mkdir(exist_ok=True)

A = np.array([[6.0, 2.0], [-3.0, 0.0]])
B = np.array([[0.0], [3.0]])
Bf = np.array([[4.0], [1.0]])
Q = np.array([[2.0, 0.0], [0.0, 7.0]])


def solve_P(gamma):
    Rt = B @ B.T - (Bf @ Bf.T)/gamma**2
    Z = np.block([[A, -Rt], [-Q, -A.T]])
    w, Vz = np.linalg.eig(Z)
    order = np.argsort(w.real)
    stable = order[:2]
    if w[stable].real.max() > -1e-6:
        return None
    U = Vz[:, stable]
    U1 = U[:2, :]
    U2 = U[2:, :]
    P = (U2 @ np.linalg.inv(U1)).real
    P = (P + P.T)/2
    if np.min(np.linalg.eigvalsh(P)) < -1e-6:
        return None
    Acl = A - B @ B.T @ P
    if np.max(np.linalg.eigvals(Acl).real) > -1e-9:
        return None
    return P


gammas = np.linspace(20.0, 0.5, 400)
gamma_hi = None
gamma_lo = None
P = None
for g in gammas:
    Pg = solve_P(g)
    if Pg is None:
        gamma_lo = g
        break
    gamma_hi = g
    P = Pg

gamma_tol = 1e-4
if gamma_lo is not None:
    while gamma_hi - gamma_lo > gamma_tol:
        mid = (gamma_lo + gamma_hi)/2
        Pm = solve_P(mid)
        if Pm is None:
            gamma_lo = mid
        else:
            gamma_hi = mid
            P = Pm

gamma_min = gamma_hi
gamma_design = gamma_min*1.01
P = solve_P(gamma_design)
K = -B.T @ P
Acl = A + B @ K
eig_cl = np.linalg.eigvals(Acl)
ts = np.linspace(0, 10, 2000)
x0 = np.array([1.0, 0.0])


def dist(t):
    return 10*np.sin(6*t) + 5*np.cos(2*t) + 4*np.cos(3*t) + 3*np.cos(8*t)


def f(t, x):
    return Acl @ x + Bf.flatten()*dist(t)


sol = solve_ivp(f, [0, 10], x0, t_eval=ts, rtol=1e-8, atol=1e-10)
X = sol.y
U = (K @ X).flatten()

plt.figure()
plt.plot(ts, X[0], label="x1"); plt.plot(ts, X[1], label="x2")
plt.xlabel("t"); plt.legend(); plt.title("H-inf closed loop states under disturbance")
plt.savefig(plots / "states.png", dpi=130, bbox_inches="tight"); plt.close()

plt.figure()
plt.plot(ts, U); plt.xlabel("t"); plt.ylabel("u"); plt.title("H-inf control")
plt.savefig(plots / "control.png", dpi=130, bbox_inches="tight"); plt.close()


def hinf_norm(Cm):
    ws = np.logspace(-2, 3, 4000)
    vals = []
    for w in ws:
        Gv = Cm @ np.linalg.solve(1j*w*np.eye(2) - Acl, Bf)
        vals.append(np.linalg.norm(Gv, 2))
    return float(np.max(vals)), ws, np.array(vals)


C1 = np.array([[1.0, 0.0]])
C2 = np.array([[0.0, 1.0]])
n1, ws, v1 = hinf_norm(C1)
n2, _, v2 = hinf_norm(C2)
nfull, _, vf = hinf_norm(np.eye(2))

plt.figure()
plt.semilogx(ws, v1, label="C1 channel")
plt.semilogx(ws, v2, label="C2 channel")
plt.semilogx(ws, vf, label="full")
plt.xlabel("omega"); plt.ylabel("|G(jw)|"); plt.legend(); plt.title("Frequency response magnitudes")
plt.savefig(plots / "freq.png", dpi=130, bbox_inches="tight"); plt.close()

res = {"gamma_min": float(gamma_min), "gamma_tol": gamma_tol,
       "gamma_design": float(gamma_design), "K": K.flatten().tolist(),
       "closed_loop_eig_real": [complex(e).real for e in eig_cl],
       "hinf_C1": n1, "hinf_C2": n2, "hinf_full": nfull}
(here / "results.json").write_text(json.dumps(res, indent=2))
print(res)
