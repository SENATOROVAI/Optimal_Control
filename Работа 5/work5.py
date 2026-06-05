import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
from pathlib import Path

here = Path(__file__).resolve().parent
plots = here / "plots"
plots.mkdir(exist_ok=True)

A = np.array([[0.0, -4.0], [1.0, -4.0]])
b = np.array([[8.0], [7.0]])
C = np.array([[1.0, 0.0]])
G = np.eye(2)
W = np.array([[4.0, 3.5], [3.5, 5.0]])
V = 5.0
x0 = np.array([1.0, 0.0])
T = 10.0
ts = np.linspace(0, T, 1000)


def observer_gain(W, V):
    P = solve_continuous_are(A.T, C.T, G @ W @ G.T, np.array([[V]]))
    L = P @ C.T / V
    return L


def simulate(L):
    def f(t, z):
        x = z[:2]; xh = z[2:]
        u = np.sin(t)
        y = float((C @ x).item())
        dx = A @ x + b.flatten()*u
        dxh = A @ xh + b.flatten()*u + L.flatten()*(y - float((C @ xh).item()))
        return np.concatenate([dx, dxh])
    z0 = np.concatenate([x0, np.zeros(2)])
    sol = solve_ivp(f, [0, T], z0, t_eval=ts, rtol=1e-8, atol=1e-10)
    e = sol.y[:2] - sol.y[2:]
    return e


L = observer_gain(W, V)
e = simulate(L)

plt.figure()
plt.plot(ts, e[0], label="e_H1"); plt.plot(ts, e[1], label="e_H2")
plt.xlabel("t"); plt.ylabel("estimation error"); plt.legend(); plt.title("Kalman filter errors")
plt.savefig(plots / "errors.png", dpi=130, bbox_inches="tight"); plt.close()

L2 = L*1.5
e2 = simulate(L2)
plt.figure()
plt.plot(ts, e[0], label="optimal L (e1)"); plt.plot(ts, e2[0], label="changed L (e1)")
plt.xlabel("t"); plt.ylabel("e1"); plt.legend(); plt.title("Effect of changed L")
plt.savefig(plots / "vary_L.png", dpi=130, bbox_inches="tight"); plt.close()

W2 = np.array([[8.0, 3.5], [3.5, 10.0]])
e3 = simulate(observer_gain(W2, V))
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.plot(ts, e[0], label="base W"); plt.plot(ts, e3[0], label="changed W"); plt.xlabel("t"); plt.ylabel("e_H1"); plt.legend(); plt.title("e_H1")
plt.subplot(1, 2, 2); plt.plot(ts, e[1], label="base W"); plt.plot(ts, e3[1], label="changed W"); plt.xlabel("t"); plt.ylabel("e_H2"); plt.legend(); plt.title("e_H2")
plt.tight_layout()
plt.savefig(plots / "vary_W.png", dpi=130, bbox_inches="tight"); plt.close()

e4 = simulate(observer_gain(W, 1.0))
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.plot(ts, e[0], label="base V"); plt.plot(ts, e4[0], label="changed V"); plt.xlabel("t"); plt.ylabel("e_H1"); plt.legend(); plt.title("e_H1")
plt.subplot(1, 2, 2); plt.plot(ts, e[1], label="base V"); plt.plot(ts, e4[1], label="changed V"); plt.xlabel("t"); plt.ylabel("e_H2"); plt.legend(); plt.title("e_H2")
plt.tight_layout()
plt.savefig(plots / "vary_V.png", dpi=130, bbox_inches="tight"); plt.close()

Qc = np.array([[1.0, 0.0], [0.0, 4.0]])
rc = 5.0
Pc = solve_continuous_are(A, b, Qc, np.array([[rc]]))
K = (b.T @ Pc) / rc


def simulate_lqg(L, K):
    def f(t, z):
        x = z[:2]; xh = z[2:]
        u = float((-K @ xh).item())
        y = float((C @ x).item())
        dx = A @ x + b.flatten()*u
        dxh = A @ xh + b.flatten()*u + L.flatten()*(y - float((C @ xh).item()))
        return np.concatenate([dx, dxh])
    z0 = np.concatenate([x0, np.zeros(2)])
    sol = solve_ivp(f, [0, T], z0, t_eval=ts, rtol=1e-8, atol=1e-10)
    return sol.y[:2], -(K @ sol.y[2:]).flatten()


Xl, Ul = simulate_lqg(L, K)
plt.figure()
plt.plot(ts, Xl[0], label="x1"); plt.plot(ts, Xl[1], label="x2"); plt.plot(ts, Ul, label="u")
plt.xlabel("t"); plt.legend(); plt.title("LQG closed loop")
plt.savefig(plots / "lqg.png", dpi=130, bbox_inches="tight"); plt.close()

res = {"L": L.flatten().tolist(), "K": K.flatten().tolist(),
       "Acl_lqg_eig_real": [complex(v).real for v in np.linalg.eigvals(A - b @ K)]}
(here / "results.json").write_text(json.dumps(res, indent=2))
print(res)
