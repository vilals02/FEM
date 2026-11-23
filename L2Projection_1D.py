import FEMAssemblers

import numpy as np
import matplotlib.pyplot as plt
def foo1(x):
    return x * np.sin(x)

def l2_projector_1d():
    n = 10                              # number of subintervals
    x = np.linspace(0, 1, n + 1)       # mesh
    M = FEMAssemblers.mass_assembler_1d(x)           # assemble mass
    b = FEMAssemblers.load_assembler_1d(x, foo1)     # assemble load
    Pf = np.linalg.solve(M, b)         # solve linear system
    plt.plot(x, Pf,'*--', label = 'Pf(x)')                    # plot L^2 projection
    xx = np.linspace(0, 1, 200)
    plt.plot(xx, foo1(xx), '--', label='f(x)')
    plt.legend()
    plt.show()

l2_projector_1d()