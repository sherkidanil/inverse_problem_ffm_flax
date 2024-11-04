import numpy as np
from typing import List
from tqdm import tqdm
import os

# from solver import pde_solution
# from kl_extension import KLExpansion
from utils import get_d_from_u

from glob import glob
from natsort import natsorted

datadir = 'data/point3_100k'
soldir = f'{datadir}/sols'

files = natsorted(glob(soldir+'/*.npy'))
m_arr = np.load(f'{datadir}/m.npy')
e_arr = np.load(f'{datadir}/e.npy')

points = [(16, 16),
          (16, 48),
          (32, 32),
          (48, 16),
          (48, 48)
         ]
d_arr = np.zeros((m_arr.shape[0], len(points)), dtype='float')

for i, file in tqdm(enumerate(files), total=len(files)):
    m = m_arr[i]
    e = e_arr[i]
    log_kappa, u = np.load(file)
    d = get_d_from_u(u, points)
    d_arr[i] = d

np.save(f'{datadir}/d_5points.npy', d_arr)