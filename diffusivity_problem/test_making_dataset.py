import numpy as np
import solver
from tqdm import tqdm

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

for i in tqdm(range(fields.shape[0])):
    e1, e2 = e_arr[i]
    u = np.zeros((Nx, Ny))
    u[ 0, : ] = np.exp(-0.5 * (y - e1) ** 2)
    u[ -1, : ] = -np.exp(-0.5 * (y - e2) ** 2)
    kappa = np.exp(fields[i][0])
    u, _ = solver.solve(u, kappa, dx, dy, omega, max_iterations, tolerance)
    solutions[i][0] = u


np.save(f'{folder}/solutions_100k.npy', solutions)
    
