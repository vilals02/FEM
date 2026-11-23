import numpy as np
import FEMAssemblers as fa
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl

def f(x):
    f = np.where(np.abs(x) <= 0.1, -5, x)
    f = np.where(np.abs(0.2 - np.abs(f)) <= 0.1, 25, f)
    f = np.where(np.abs(0.6 - np.abs(f)) <= 0.1, -30, f)
    f = np.where(np.abs(0.9 - np.abs(f)) <= 0.2, 20, f)
    f = np.where(np.abs(f) <= 1, 1, f)

    return f

def deltafunc(x):
    delta = 0.01
    
    return delta

def eta2(laplace_Uh, f, x):

    delta = 0.01

    intervals = len(laplace_Uh) - 1
    out = np.zeros(intervals)

    for i in range(intervals):

        hi = x[i+1] - x[i]

        fip = (f[i+1] + delta*laplace_Uh[i+1])**2

        fi = (f[i] + delta*laplace_Uh[i])**2

        out[i] = 0.5*hi*(fip + fi)*(hi**2)
        
    return out

def refine_grid(etas2, x):
    lam = 0.9

    for i in range(len(etas2)):
        max_eta2 = np.max(etas2)
        if etas2[i] > lam*max_eta2:
            x = np.append(x, (x[i+1] + x[i])/2)

    x = np.sort(x)

    return x



kappa = np.array([1e6, 1e6])
delta = 0.01
tol = 1e-3
n = 12
xgrid = np.linspace(-1, 1, n)
etas2 = np.ones(n-1)

while n < 1e4:
    A = fa.stiffness_assembler_1d(xgrid, deltafunc, kappa)
    b = fa.load_assembler_1d(xgrid, f)
    Uh = sp.sparse.linalg.spsolve(A, b)

    M = fa.mass_assembler_1d(xgrid)

    w = np.zeros(len(xgrid))
    w[1:-1] = -np.linalg.solve(M[1:-1, 1:-1], A[1:-1, 1:-1] @ Uh[1:-1])/delta
    w[0], w[-1] = w[1], w[-2]        

    etas2 = eta2(w, f(xgrid), xgrid)

    if np.sum(etas2) < tol:
        break

    xgrid = refine_grid(etas2, xgrid)
    n = len(xgrid)

residuals = f(xgrid) + delta*w

mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",        
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "lines.linewidth": 1.6,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# quantities per element (midpoints), so eta and 1/h sit where they belong
xm = 0.5 * (xgrid[:-1] + xgrid[1:])
h = np.diff(xgrid)
jumps = [-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7]

fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

# (a) solution, with the mesh nodes shown as markers
ax[0, 0].plot(xgrid, Uh, color="C0", marker="o", ms=2.5, mew=0)
ax[0, 0].set_ylabel(r"$U_h(x)$")
ax[0, 0].set_title(r"(a) Finite element solution")

# (b) residual at the nodes
ax[0, 1].plot(xgrid, residuals, color="C3")
ax[0, 1].axhline(0, color="k", lw=0.8)
ax[0, 1].set_ylabel(r"$R(U_h) = f + \delta\,\Delta_h U_h$")
ax[0, 1].set_title(r"(b) Residual")

# (c) element error indicators
ax[1, 0].plot(xm, np.sqrt(etas2), color="C2")
ax[1, 0].set_ylabel(r"$\eta_i$")
ax[1, 0].set_title(r"(c) Error indicators")

# (d) local mesh density
ax[1, 1].plot(xm, 1 / h, color="C1")
ax[1, 1].set_ylabel(r"$1/h_i$")
ax[1, 1].set_title(r"(d) Mesh density")

for a in ax.ravel():
    for k, xj in enumerate(jumps):               # discontinuities of f
        a.axvline(xj, color="gray", lw=1, ls=":", zorder=0,
                  label=r"jumps of $f$" if (k == 0 and a is ax[0, 1]) else None)
    a.set_xlim(-1, 1)
for a in ax[1, :]:
    a.set_xlabel(r"$x$")
ax[0, 1].legend(loc="upper right", frameon=True)

fig.suptitle(rf"Adaptive FEM, $\delta = 0.01$, $N = {len(xgrid)}$ nodes", y=1.01)
fig.tight_layout()
fig.savefig("adaptive_fem.pdf")                  # vector format for reports
plt.show()



