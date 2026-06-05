import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

here = Path(__file__).resolve().parent
plots = here / "plots"
plots.mkdir(exist_ok=True)

H = np.array([[4.0, 5.0], [5.0, 12.0]])
g0 = np.array([10.0, 1.0])


def J(x, u):
    return 2*x**2 + 6*u**2 + 5*x*u + 10*x + u - 5


def grad(z):
    x, u = z
    return np.array([4*x + 5*u + 10, 12*u + 5*x + 1])


z_unc = np.linalg.solve(H, -g0)
eig = np.linalg.eigvalsh(H)
J_unc = J(z_unc[0], z_unc[1])

A_eq = np.array([[4.0, 5.0, -1.0], [5.0, 12.0, -2.0], [1.0, 2.0, 0.0]])
b_eq = np.array([-10.0, -1.0, -2.0])
sol_eq = np.linalg.solve(A_eq, b_eq)
z_eq = sol_eq[:2]
lam_eq = sol_eq[2]
J_eq = J(z_eq[0], z_eq[1])

c_unc = z_unc[0] + 2*z_unc[1] + 2
if c_unc <= 0:
    z_ineq, J_ineq, ineq_active = z_unc, J_unc, False
else:
    z_ineq, J_ineq, ineq_active = z_eq, J_eq, True

z = np.array([0.0, 0.0])
newton_path = [z.copy()]
for _ in range(2):
    z = z - np.linalg.solve(H, grad(z))
    newton_path.append(z.copy())
newton_path = np.array(newton_path)


def steepest(gamma, n=60):
    z = np.array([0.0, 0.0])
    js = [J(z[0], z[1])]
    path = [z.copy()]
    for _ in range(n):
        z = z - gamma*grad(z)
        js.append(J(z[0], z[1]))
        path.append(z.copy())
    return np.array(js), z, np.array(path)


g_aper = 0.05
g_osc = 0.13
js_aper, z_aper, path_aper = steepest(g_aper)
js_osc, z_osc, path_osc = steepest(g_osc)
steps_aper = path_aper[:7]
steps_osc = path_osc[:7]

xs = np.linspace(-12, 4, 200)
us = np.linspace(-4, 6, 200)
XX, UU = np.meshgrid(xs, us)
ZZ = J(XX, UU)

plt.figure()
cs = plt.contour(XX, UU, ZZ, 30)
plt.plot(z_unc[0], z_unc[1], "ro", label="unconstrained min")
plt.plot(z_eq[0], z_eq[1], "gs", label="equality min")
xl = np.linspace(-12, 4, 50)
plt.plot(xl, -(xl+2)/2, "k--", label="c(x,u)=0")
plt.xlabel("x"); plt.ylabel("u"); plt.legend(); plt.title("Cost contours and minima")
plt.savefig(plots / "contours.png", dpi=130, bbox_inches="tight"); plt.close()

plt.figure()
plt.plot(js_aper, label=f"gamma={g_aper} (aperiodic)")
plt.plot(js_osc, label=f"gamma={g_osc} (oscillatory)")
plt.axhline(J_unc, color="k", ls=":", label="J min")
plt.xlabel("iteration"); plt.ylabel("J"); plt.legend(); plt.title("Steepest descent convergence")
plt.savefig(plots / "steepest.png", dpi=130, bbox_inches="tight"); plt.close()

res = {
    "Hessian": H.tolist(),
    "Hessian_eigs": eig.tolist(),
    "z_unc": z_unc.tolist(), "J_unc": float(J_unc),
    "z_eq": z_eq.tolist(), "lambda_eq": float(lam_eq), "J_eq": float(J_eq),
    "ineq_active": bool(ineq_active), "z_ineq": z_ineq.tolist(), "J_ineq": float(J_ineq),
    "newton_path": newton_path.tolist(),
    "gamma_aper": g_aper, "gamma_osc": g_osc,
    "z_aper": z_aper.tolist(), "z_osc": z_osc.tolist(),
    "steps_aper": steps_aper.tolist(), "J_aper_steps": js_aper[:7].tolist(),
    "steps_osc": steps_osc.tolist(), "J_osc_steps": js_osc[:7].tolist(),
}
(here / "results.json").write_text(json.dumps(res, indent=2))
print(res)
