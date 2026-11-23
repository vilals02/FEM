import numpy as np
import numpy as np
from scipy . spatial import Delaunay
from scipy.sparse import coo_matrix

def hat_gradients(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    detJ = ((x[1] - x[0]) * (y[2] - y[0])
            - (x[2] - x[0]) * (y[1] - y[0]))

    if abs(detJ) < 1.0e-15:
        raise ValueError("Degenerate triangle.")

    area = 0.5 * abs(detJ)

    b = np.array([y[1] - y[2],
                  y[2] - y[0],
                  y[0] - y[1]], dtype=float) / detJ

    c = np.array([x[2] - x[1],
                  x[0] - x[2],
                  x[1] - x[0]], dtype=float) / detJ

    return area, b, c

def _assemble_sparse(npnt, t, local_matrix):
    nt = t.shape[1] # Iterate through triangles 

    rows = []
    cols = []
    vals = []

    for K in range(nt):
        loc2glb = t[:, K]
        AK = local_matrix(K, loc2glb)

        for i in range(3):
            for j in range(3):
                rows.append(loc2glb[i])
                cols.append(loc2glb[j])
                vals.append(AK[i, j])

    return coo_matrix((vals, (rows, cols)),
                      shape=(npnt, npnt)).tocsr()

def _assemble_stiffness_sparse(npnt, p, t, local_matrix):
    nt = t.shape[1] # Iterate through triangles 

    rows = []
    cols = []
    vals = []

    for K in range(nt):
        loc2glb = t[:, K]
        AK = local_matrix(K, loc2glb)

        for i in range(3):
            for j in range(3):
                rows.append(loc2glb[i])
                cols.append(loc2glb[j])
                vals.append(AK[i, j])

    return coo_matrix((vals, (rows, cols)),
                      shape=(npnt, npnt)).tocsr()

def stiffness_assembler_2d(p, t):
    """
    Assemble A_ij = int a grad(phi_j).grad(phi_i) dx.

    The coefficient a is evaluated at the triangle centroid,
    as in Larson-Bengzon.
    """
    npnt = p.shape[1]

    def local_matrix(K, loc2glb):
        x = p[0, loc2glb]
        y = p[1, loc2glb]

        area, b, c = hat_gradients(x, y)
        xc = np.mean(x)
        yc = np.mean(y)
        # abar = float(a(xc, yc)) # Larson-Bengzon

        return area * (np.outer(b, b) + np.outer(c, c))

    return _assemble_stiffness_sparse(npnt, p, t, local_matrix)

def load_assembler_2d(p, t, f):
    """
    Assemble b_i = int f phi_i dx using the corner quadrature rule
    from Larson-Bengzon.
    """
    npnt = p.shape[1]
    nt = t.shape[1]

    b_global = np.zeros(npnt)

    for K in range(nt):
        loc2glb = t[:, K]

        x = p[0, loc2glb]
        y = p[1, loc2glb]

        area, _, _ = hat_gradients(x, y)

        bK = area / 3.0 * np.array([
            f(x[0], y[0]),
            f(x[1], y[1]),
            f(x[2], y[2])
        ], dtype=float)

        b_global[loc2glb] += bK

    return b_global


def circle_mesh(hmax):
    # Nodes on concentric circles with spacing about hmax
    pts = [np.zeros((1, 2))]
    nr = int(np.ceil(1 / hmax))

    for k in range(1, nr + 1):
        r = k / nr
        n = int(np.ceil(2 * np.pi * r / hmax))
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts.append(np.column_stack((r * np.cos(th), r * np.sin(th))))

    p = np.vstack(pts)
    t = Delaunay(p).simplices
    e = np.flatnonzero(np.isclose(np.hypot(p[:, 0], p[:, 1]), 1.0))

    return p.T, e, t.T

def mass_assembler_1d(x):
    n = len(x) - 1                  # number of subintervals
    M = np.zeros((n + 1, n + 1))    # allocate mass matrix
    for i in range(n):              # loop over subintervals
        h = x[i + 1] - x[i]         # interval length
        M[i, i] += h / 3
        M[i, i + 1] += h / 6
        M[i + 1, i] += h / 6
        M[i + 1, i + 1] += h / 3
    return M


def load_assembler_1d(x, f):
    n = len(x) - 1
    b = np.zeros(n + 1)
    for i in range(n):
        h = x[i + 1] - x[i]
        b[i] += f(x[i]) * h / 2
        b[i + 1] += f(x[i + 1]) * h / 2
    return b

import numpy as np

def stiffness_assembler_1d(x, a, kappa):
    n = len(x) - 1
    A = np.zeros((n + 1, n + 1))
    for i in range(n):
        h = x[i + 1] - x[i]
        xmid = (x[i + 1] + x[i]) / 2   # interval mid-point
        amid = a(xmid)                 # value of a(x) at mid-point
        A[i, i] += amid / h
        A[i, i + 1] -= amid / h
        A[i + 1, i] -= amid / h
        A[i + 1, i + 1] += amid / h
    A[0, 0] += kappa[0]
    A[n, n] += kappa[1]
    return A

def source_assembler_1d(x, f, kappa, g):
    b = load_assembler_1d(x, f)
    b[0] += kappa[0] * g[0]
    b[-1] += kappa[1] * g[1]
    return b
