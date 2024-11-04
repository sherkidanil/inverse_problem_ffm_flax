# -div (\kappa \nabla u) = 0 
# kappa is custom defined 2d grid 
# u(x = 0, y) = f(y, e1) = exp(-0.5 * (y - e1)^2)
#  u(x = 1, y) = g(y, e2) = - exp(-0.5 * (y - e2)^2)

import numpy as np
from numba import njit
import matplotlib.pyplot as plt

from typing import Sequence


def pde_solution(m: np.array, e: Sequence[float], omega: float = 1.5,
                 max_iterations: int = 100_000, tolerance: float = 1e-6,
                 verbose: bool = True):
    e1, e2 = e
    Nx, Ny = m.shape
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    dx = x[ 1 ] - x[ 0 ]
    dy = y[ 1 ] - y[ 0 ]
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.zeros((Nx, Ny))
    # Boundary conditions
    u[ 0, : ] = np.exp(-0.5 * (y - e1) ** 2)
    u[ -1, : ] = -np.exp(-0.5 * (y - e2) ** 2)

    kappa = np.exp(m) # kappa.shape == (Nx, Ny)

    u, diff = solve(u, kappa, dx, dy, omega, max_iterations, tolerance)
    if verbose:
        print(diff)
    return u




@njit
def solve(u, kappa, dx, dy, omega, max_iterations, tolerance):
    Nx, Ny = u.shape
    for iteration in range(max_iterations):
        u_old = u.copy()

        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                kappa_iphalf_j = 0.5 * (kappa[ i, j ] + kappa[ i + 1, j ])
                kappa_imhalf_j = 0.5 * (kappa[ i, j ] + kappa[ i - 1, j ])
                kappa_i_jphalf = 0.5 * (kappa[ i, j ] + kappa[ i, j + 1 ])
                kappa_i_jmhalf = 0.5 * (kappa[ i, j ] + kappa[ i, j - 1 ])

                A_ij = (kappa_iphalf_j + kappa_imhalf_j) / dx ** 2 + \
                       (kappa_i_jphalf + kappa_i_jmhalf) / dy ** 2

                B_ij = (kappa_iphalf_j * u[ i + 1, j ] + kappa_imhalf_j * u[ i - 1, j ]) / dx ** 2 + \
                       (kappa_i_jphalf * u[ i, j + 1 ] + kappa_i_jmhalf * u[ i, j - 1 ]) / dy ** 2

                u_new = B_ij / A_ij
                u[ i, j ] = (1 - omega) * u[ i, j ] + omega * u_new

        j = 0
        for i in range(1, Nx - 1):
            kappa_iphalf_j = 0.5 * (kappa[ i, j ] + kappa[ i + 1, j ])
            kappa_imhalf_j = 0.5 * (kappa[ i, j ] + kappa[ i - 1, j ])
            kappa_i_jphalf = 0.5 * (kappa[ i, j ] + kappa[ i, j + 1 ])
            kappa_i_jmhalf = kappa[ i, j ]

            A_ij = (kappa_iphalf_j + kappa_imhalf_j) / dx ** 2 + \
                   (kappa_i_jphalf + kappa_i_jmhalf) / dy ** 2

            B_ij = (kappa_iphalf_j * u[ i + 1, j ] + kappa_imhalf_j * u[ i - 1, j ]) / dx ** 2 + \
                   (kappa_i_jphalf * u[ i, j + 1 ] + kappa_i_jmhalf * u[ i, j ]) / dy ** 2

            u_new = B_ij / A_ij
            u[ i, j ] = (1 - omega) * u[ i, j ] + omega * u_new

        j = Ny - 1
        for i in range(1, Nx - 1):
            kappa_iphalf_j = 0.5 * (kappa[ i, j ] + kappa[ i + 1, j ])
            kappa_imhalf_j = 0.5 * (kappa[ i, j ] + kappa[ i - 1, j ])
            kappa_i_jphalf = kappa[ i, j ]
            kappa_i_jmhalf = 0.5 * (kappa[ i, j ] + kappa[ i, j - 1 ])

            A_ij = (kappa_iphalf_j + kappa_imhalf_j) / dx ** 2 + \
                   (kappa_i_jphalf + kappa_i_jmhalf) / dy ** 2

            B_ij = (kappa_iphalf_j * u[ i + 1, j ] + kappa_imhalf_j * u[ i - 1, j ]) / dx ** 2 + \
                   (kappa_i_jphalf * u[ i, j ] + kappa_i_jmhalf * u[ i, j - 1 ]) / dy ** 2

            u_new = B_ij / A_ij
            u[ i, j ] = (1 - omega) * u[ i, j ] + omega * u_new

        diff = np.linalg.norm(u - u_old)
        if diff < tolerance:
            break
    return u,  diff



def viasualize(X, Y, kappa, u, file=None):
    plt.figure(figsize=(12, 4), dpi=200)
    plt.subplot(121)
    plt.contourf(X, Y, np.log(kappa), levels=50, cmap='viridis')
    plt.colorbar()
    plt.title(fr'$\log \kappa$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(122)
    plt.contourf(X, Y, u, levels=50, cmap='viridis')
    plt.colorbar(label='u(x,y)')
    plt.title(fr'Solution of $-\nabla \cdot (\kappa \nabla u) = 0$')
    plt.xlabel('x')
    plt.ylabel('y')
    if file is not None:
        plt.savefig(file)
    plt.show()
