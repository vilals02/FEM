import numpy as np
import FEMAssemblers as fa
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
import matplotlib.tri as mtri



def f(x, y):

    return 8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

def u_exact(x, y):
    return np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

def compute_p(EnEs):
    sum = 0
    for i in range(len(EnEs) - 1):
        sum += np.log(EnEs[i]/EnEs[i+1])

    return sum/(len(EnEs) - 1)/np.log(2)



EnEs = np.zeros(5)

hmaxes = np.array([1/2, 1/4, 1/8, 1/16, 1/32])
i = 0
sols = {}

for hmax in hmaxes:

    p, e, t = fa.circle_mesh(hmax)

    Ao = fa.stiffness_assembler_2d(p, t)
    b = fa.load_assembler_2d(p, t, f)
    A = Ao.tolil()
    A[e,:] = 0
    A[e, e] = 1

    A = A.tocsr()
    b[e] = u_exact(p[0, e], p[1, e])

    uh = sp.sparse.linalg.spsolve(A, b)
    if hmax in (hmaxes[0], hmaxes[-1]):
        sols[hmax] = (p, t, uh)

    u = u_exact(p[0, :], p[1, :])

    err = u - uh

    EnE = np.sqrt(err@Ao@err)

    EnEs[i] = EnE
    i += 1

CR = np.polyfit(np.log(hmaxes), np.log(EnEs), 1)[0]
rates = np.log(EnEs[:-1] / EnEs[1:]) / np.log(hmaxes[:-1] / hmaxes[1:])

print(f"{'hmax':>8} {'energy error':>14} {'rate':>7}")
for k in range(len(hmaxes)):
    r = f"{rates[k-1]:7.3f}" if k > 0 else f"{'-':>7}"
    print(f"{hmaxes[k]:8.5f} {EnEs[k]:14.6e} {r}")
print(f"\nEstimated convergence rate (least-squares slope): p = {CR:.3f}")

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.loglog(hmaxes, EnEs, 'o-', lw=1.5, label=r'$\|u-u_h\|_E$')
ax.loglog(hmaxes, EnEs[0] * (hmaxes / hmaxes[0])**CR, '--', color='gray',
          label=rf'$h_{{\max}}^{{p}}$, $p={CR:.2f}$ (scaled)')

# x ticks exactly at the mesh sizes, labeled 1/2, 1/4, ...
ax.xaxis.set_major_locator(FixedLocator(hmaxes))
ax.xaxis.set_major_formatter(FixedFormatter([f'1/{int(round(1/h))}' for h in hmaxes]))
ax.xaxis.set_minor_formatter(NullFormatter())   # remove the crowded minor labels

ax.set_xlabel(r'$h_{\max}$')
ax.set_ylabel('energy norm of the error')
ax.grid(True, which='major', ls=':', alpha=0.6)
ax.legend()
fig.tight_layout()
fig.savefig('convergence.png', dpi=200)
plt.show()

fig = plt.figure(figsize=(11, 4.8))

for k, hm in enumerate((hmaxes[0], hmaxes[-1])):
    pp, tt, uu = sols[hm]
    triang = mtri.Triangulation(pp[0], pp[1], tt.T)

    ax = fig.add_subplot(1, 2, k + 1, projection='3d')
    surf = ax.plot_trisurf(triang, uu, cmap='viridis',
                           edgecolor='k' if hm == hmaxes[0] else 'none',
                           linewidth=0.3, antialiased=True)
    ax.set_title(rf'$h_{{\max}}={hm:g}$')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_zlabel(r'$u_h$')
    ax.set_zlim(-1.1, 1.1)
    ax.view_init(elev=30, azim=-60)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)

fig.tight_layout()
fig.savefig('solutions_3d.png', dpi=200)
plt.show()