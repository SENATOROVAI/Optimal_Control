import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from pathlib import Path

here = Path(__file__).resolve().parent
plots = here / "plots"
plots.mkdir(exist_ok=True)

A = np.array([[0.0, 1.0], [-6.0, -5.0]])
b = np.array([[0.0], [1.0]])
Q = np.array([[1.0, 0.0], [0.0, 4.0]])
r = 5.0
x0 = np.array([1.0, 0.0])
T = 5.0
ts = np.linspace(0, T, 600)


def riccati_rhs(t, p):
    P = p.reshape(2, 2)
    dP = -(A.T @ P + P @ A - P @ b @ b.T @ P / r + Q)
    return dP.flatten()


solP = solve_ivp(riccati_rhs, [T, 0], np.zeros(4), t_eval=ts[::-1], rtol=1e-8, atol=1e-10)
Pgrid = solP.y[:, ::-1]


def Pat(t):
    i = np.searchsorted(ts, t)
    i = min(max(i, 0), ts.size-1)
    return Pgrid[:, i].reshape(2, 2)


def simulate(scale=1.0):
    def f(t, x):
        K = scale*(b.T @ Pat(t))/r
        u = float((-K @ x).item())
        return A @ x + b.flatten()*u
    sol = solve_ivp(f, [0, T], x0, t_eval=ts, rtol=1e-8, atol=1e-10)
    X = sol.y
    U = np.array([float((-(b.T @ Pat(t))/r @ X[:, i]).item())*scale for i, t in enumerate(ts)])
    integ = X[0]**2*Q[0, 0] + X[1]**2*Q[1, 1] + r*U**2
    Jt = np.concatenate([[0], cumulative_trapezoid(integ, ts)])
    return X, U, Jt


X, U, Jt = simulate(1.0)

plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1); plt.plot(ts, X[0]); plt.xlabel("t"); plt.ylabel("x1"); plt.title("x1")
plt.subplot(2, 2, 2); plt.plot(ts, X[1]); plt.xlabel("t"); plt.ylabel("x2"); plt.title("x2")
plt.subplot(2, 2, 3); plt.plot(ts, U); plt.xlabel("t"); plt.ylabel("u"); plt.title("u")
plt.subplot(2, 2, 4); plt.plot(ts, Jt); plt.xlabel("t"); plt.ylabel("J"); plt.title("J")
plt.tight_layout()
plt.savefig(plots / "optimal.png", dpi=130, bbox_inches="tight"); plt.close()

plt.figure()
plt.plot(ts, Jt, label="optimal")
for s in [0.8, 1.2]:
    _, _, Js = simulate(s)
    plt.plot(ts, Js, label=f"gain x{s}")
plt.xlabel("t"); plt.ylabel("J(t)"); plt.legend(); plt.title("Cost: optimal vs perturbed")
plt.savefig(plots / "perturbed.png", dpi=130, bbox_inches="tight"); plt.close()

res = {"P0": Pat(0.0).tolist(), "J_opt": float(Jt[-1]),
       "K0": (b.T @ Pat(0.0) / r).flatten().tolist()}
(here / "results.json").write_text(json.dumps(res, indent=2))
print(res)
