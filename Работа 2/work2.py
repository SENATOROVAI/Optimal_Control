import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, cumulative_trapezoid
from pathlib import Path

here = Path(__file__).resolve().parent
plots = here / "plots"
plots.mkdir(exist_ok=True)

r = 5.0
T = 4.0


def odes(t, y):
    x1, x2, l1, l2 = y
    u = -l2/(2*r)
    return np.vstack([x2,
                      -6*x1 - 5*x2 + u,
                      -2*x1 + 6*l2,
                      -8*x2 - l1 + 5*l2])


def bc(ya, yb):
    return np.array([ya[0], ya[1], yb[0]-1.0, yb[1]])


t = np.linspace(0, T, 200)
y0 = np.zeros((4, t.size))
y0[0] = t/T
sol = solve_bvp(odes, bc, t, y0, max_nodes=20000, tol=1e-6)
ts = np.linspace(0, T, 400)
Y = sol.sol(ts)
x1, x2, l1, l2 = Y
u = -l2/(2*r)
integrand = x1**2 + 4*x2**2 + 5*u**2
Jt = np.concatenate([[0], cumulative_trapezoid(integrand, ts)])
J_opt = Jt[-1]


def cost_for_r(rd):
    def od(t, y):
        x1, x2, l1, l2 = y
        u = -l2/(2*rd)
        return np.vstack([x2, -6*x1 - 5*x2 + u, -2*x1 + 6*l2, -8*x2 - l1 + 5*l2])
    s = solve_bvp(od, bc, t, y0, max_nodes=20000, tol=1e-6)
    Yr = s.sol(ts)
    x1r, x2r, l2r = Yr[0], Yr[1], Yr[3]
    ur = -l2r/(2*rd)
    integ = x1r**2 + 4*x2r**2 + 5*ur**2
    return float(cumulative_trapezoid(integ, ts)[-1])


rs = np.linspace(3.0, 7.0, 9)
Jr = np.array([cost_for_r(v) for v in rs])

plt.figure()
plt.plot(ts, x1, label="x1"); plt.plot(ts, x2, label="x2")
plt.xlabel("t"); plt.ylabel("state"); plt.legend(); plt.title("Optimal state trajectory")
plt.savefig(plots / "state.png", dpi=130, bbox_inches="tight"); plt.close()

plt.figure()
plt.plot(ts, u); plt.xlabel("t"); plt.ylabel("u"); plt.title("Optimal control")
plt.savefig(plots / "control.png", dpi=130, bbox_inches="tight"); plt.close()

plt.figure()
plt.plot(ts, Jt); plt.xlabel("t"); plt.ylabel("J(t)"); plt.title("Accumulated cost")
plt.savefig(plots / "cost.png", dpi=130, bbox_inches="tight"); plt.close()

plt.figure()
plt.plot(rs, Jr, "o-"); plt.axvline(5.0, color="k", ls=":")
plt.xlabel("design weight r"); plt.ylabel("nominal cost J"); plt.title("Cost vs changed controller parameter r")
plt.savefig(plots / "cost_perturbed.png", dpi=130, bbox_inches="tight"); plt.close()

res = {"converged": bool(sol.success), "J_opt": float(J_opt),
       "u0": float(u[0]), "uT": float(u[-1]),
       "x_final": [float(x1[-1]), float(x2[-1])],
       "r_grid": rs.tolist(), "J_grid": Jr.tolist(),
       "r_best": float(rs[int(np.argmin(Jr))])}
(here / "results.json").write_text(json.dumps(res, indent=2))
print(res)
