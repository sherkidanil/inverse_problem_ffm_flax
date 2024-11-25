import numpy as np
import solver
from tqdm import tqdm
from numba import njit

from typing import Sequence

folder = '../diffusion_ffno/data/try1'
e_arr = np.load(f'{folder}/e_bnds.npy')
fields = np.load(f'{folder}/fields_100k.npy')

omega = 1.5
max_iterations = 100_000
tolerance = 1e-6

Nx, Ny = fields.shape[2], fields.shape[3]
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
dx = x[ 1 ] - x[ 0 ]
dy = y[ 1 ] - y[ 0 ]
X, Y = np.meshgrid(x, y, indexing='ij')

solutions = np.zeros_like(fields)


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


for i in tqdm(range(fields.shape[0])):
    e1, e2 = e_arr[i]
    u = np.zeros((Nx, Ny))
    u[ 0, : ] = np.exp(-0.5 * (y - e1) ** 2)
    u[ -1, : ] = -np.exp(-0.5 * (y - e2) ** 2)
    kappa = np.exp(fields[i][0])
    u, _ = solver.solve(u, kappa, dx, dy, omega, max_iterations, tolerance)
    solutions[i][0] = u


np.save(f'{folder}/solutions_100k_jax.npy', solutions)
    
