import numpy as np
from typing import List
from tqdm import tqdm
import os

from solver import pde_solution
from kl_extension import KLExpansion
from utils import get_d_from_u

datadir = 'data/point3_100k'
soldir = f'{datadir}/sols'

os.makedirs(datadir, exist_ok=True)
os.makedirs(soldir, exist_ok=True)

points = [(10, 10), (40, 40), (50, 10)]

kl = KLExpansion(grid=(64, 64))
kl.calculate_eigh()

SIZE = 100_000

m_arr = np.zeros((SIZE, 16), dtype='float')
e_arr = np.zeros((SIZE, 2), dtype='float')
d_arr = np.zeros((SIZE, 3), dtype='float')

for i in tqdm(range(SIZE)):
    m = np.random.normal(size = 16)
    e = np.random.uniform(size = 2)
    log_kappa = kl.expansion(m)
    u = pde_solution(log_kappa, e, verbose=False)
    np.save(f'{soldir}/solution_{i}.npy', np.stack([log_kappa, u]))
    d = get_d_from_u(u, points)
    m_arr[i] = m
    e_arr[i] = e
    d_arr[i] = d
    
np.save(f'{datadir}/m.npy', m_arr)
np.save(f'{datadir}/e.npy', e_arr)
np.save(f'{datadir}/d.npy', d_arr)