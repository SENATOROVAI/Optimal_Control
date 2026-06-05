import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp, cumulative_trapezoid
from pathlib import Path

here = Path(__file__).resolve().parent
plots = here / "plots"
plots.mkdir(exist_ok=True)

A = np.array([[0.0, 1.0], [-2.0, 5.0]])
b = np.array([[0.0], [3.0]])
Q = np.array([[1.0, 0.0], [0.0, 2.0]])
r = 5.0
x0 = np.array([1.0, 0.0])
T = 8.0
ts = np.linspace(0, T, 600)


def lqr(A, b, Q, r):
    P = solve_continuous_are(A, b, Q, np.array([[r]]))
    K = (b.T @ P) / r
    return P, K


def simulate(K):
    def f(t, x):
        u = float((-K @ x).item())
        return A @ x + b.flatten()*u
    sol = solve_ivp(f, [0, T], x0, t_eval=ts, rtol=1e-8, atol=1e-10)
    X = sol.y
    U = -(K @ X).flatten()
    integ = (X[0]**2*Q[0, 0] + X[1]**2*Q[1, 1] + r*U**2)
    Jt = np.concatenate([[0], cumulative_trapezoid(integ, ts)])
    return X, U, Jt


P, K = lqr(A, b, Q, r)
Acl = A - b @ K
eig_cl = np.linalg.eigvals(Acl)
X, U, Jt = simulate(K)
J_ss = Jt[-1]
J_inf = float(x0 @ P @ x0)

K2 = K*1.05
Acl2 = A - b @ K2
eig_cl2 = np.linalg.eigvals(Acl2)
X2, U2, Jt2 = simulate(K2)

plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1); plt.plot(ts, X[0]); plt.xlabel("t"); plt.ylabel("x1"); plt.title("x1")
plt.subplot(2, 2, 2); plt.plot(ts, X[1]); plt.xlabel("t"); plt.ylabel("x2"); plt.title("x2")
plt.subplot(2, 2, 3); plt.plot(ts, U); plt.xlabel("t"); plt.ylabel("u"); plt.title("u")
plt.subplot(2, 2, 4); plt.plot(ts, Jt); plt.xlabel("t"); plt.ylabel("J"); plt.title("J")
plt.tight_layout()
plt.savefig(plots / "optimal.png", dpi=130, bbox_inches="tight"); plt.close()

plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1); plt.plot(ts, X[0], label="optimal K"); plt.plot(ts, X2[0], label="perturbed K"); plt.xlabel("t"); plt.ylabel("x1"); plt.legend(); plt.title("x1")
plt.subplot(2, 2, 2); plt.plot(ts, X[1], label="optimal K"); plt.plot(ts, X2[1], label="perturbed K"); plt.xlabel("t"); plt.ylabel("x2"); plt.legend(); plt.title("x2")
plt.subplot(2, 2, 3); plt.plot(ts, U, label="optimal K"); plt.plot(ts, U2, label="perturbed K"); plt.xlabel("t"); plt.ylabel("u"); plt.legend(); plt.title("u")
plt.subplot(2, 2, 4); plt.plot(ts, Jt, label="optimal K"); plt.plot(ts, Jt2, label="perturbed K"); plt.xlabel("t"); plt.ylabel("J"); plt.legend(); plt.title("J")
plt.tight_layout()
plt.savefig(plots / "perturbed_K.png", dpi=130, bbox_inches="tight"); plt.close()

rs = [1.0, 5.0, 20.0]
plt.figure(figsize=(8, 6))
for rr in rs:
    _, Kr = lqr(A, b, Q, rr)
    Xr, Ur, Jr = simulate(Kr)
    plt.subplot(2, 2, 1); plt.plot(ts, Xr[0], label=f"r={rr}")
    plt.subplot(2, 2, 2); plt.plot(ts, Xr[1], label=f"r={rr}")
    plt.subplot(2, 2, 3); plt.plot(ts, Ur, label=f"r={rr}")
    plt.subplot(2, 2, 4); plt.plot(ts, Jr, label=f"r={rr}")
plt.subplot(2, 2, 1); plt.xlabel("t"); plt.ylabel("x1"); plt.legend(); plt.title("x1")
plt.subplot(2, 2, 2); plt.xlabel("t"); plt.ylabel("x2"); plt.legend(); plt.title("x2")
plt.subplot(2, 2, 3); plt.xlabel("t"); plt.ylabel("u"); plt.legend(); plt.title("u")
plt.subplot(2, 2, 4); plt.xlabel("t"); plt.ylabel("J"); plt.legend(); plt.title("J")
plt.tight_layout()
plt.savefig(plots / "vary_r.png", dpi=130, bbox_inches="tight"); plt.close()

ks = [0.5, 1.0, 5.0]
plt.figure(figsize=(8, 6))
for k in ks:
    _, Kq = lqr(A, b, k*Q, r)
    Xq, Uq, Jq = simulate(Kq)
    plt.subplot(2, 2, 1); plt.plot(ts, Xq[0], label=f"k={k}")
    plt.subplot(2, 2, 2); plt.plot(ts, Xq[1], label=f"k={k}")
    plt.subplot(2, 2, 3); plt.plot(ts, Uq, label=f"k={k}")
    plt.subplot(2, 2, 4); plt.plot(ts, Jq, label=f"k={k}")
plt.subplot(2, 2, 1); plt.xlabel("t"); plt.ylabel("x1"); plt.legend(); plt.title("x1")
plt.subplot(2, 2, 2); plt.xlabel("t"); plt.ylabel("x2"); plt.legend(); plt.title("x2")
plt.subplot(2, 2, 3); plt.xlabel("t"); plt.ylabel("u"); plt.legend(); plt.title("u")
plt.subplot(2, 2, 4); plt.xlabel("t"); plt.ylabel("J"); plt.legend(); plt.title("J")
plt.tight_layout()
plt.savefig(plots / "vary_Q.png", dpi=130, bbox_inches="tight"); plt.close()

res = {"P": P.tolist(), "K": K.flatten().tolist(),
       "closed_loop_eig": [complex(e).real for e in eig_cl],
       "closed_loop_eig_imag": [complex(e).imag for e in eig_cl],
       "J_ss": float(J_ss),
       "J_inf": J_inf,
       "J_ss_pert": float(Jt2[-1]),
       "K_pert": K2.flatten().tolist(),
       "closed_loop_eig_pert": [complex(e).real for e in eig_cl2]}
(here / "results.json").write_text(json.dumps(res, indent=2))
print(res)
