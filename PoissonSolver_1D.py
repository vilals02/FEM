import numpy as np
import matplotlib.pyplot as plt
import FEMAssemblers

def conductivity(x):
    return 0.1 * (5 - 0.6 * x)   # heat conductivity times area

def source(x):
    return 0.03 * (x - 6)**4     # heat source


def poisson_solver_1d():
    h = 0.1                                    # mesh size
    n = round((8 - 2) / h)                     # number of subintervals
    x = np.linspace(2, 8, n + 1)               # mesh
    kappa = [1.e+6, 0]
    g = [-1, 0]
    A = FEMAssemblers.stiffness_assembler_1d(x, conductivity, kappa)
    b = FEMAssemblers.source_assembler_1d(x, source, kappa, g)
    u = np.linalg.solve(A, b)
    plt.plot(x, u)
    plt.show()

poisson_solver_1d()