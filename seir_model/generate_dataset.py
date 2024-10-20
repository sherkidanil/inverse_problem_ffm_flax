import numpy as np
from scipy.integrate import solve_ivp
from typing import List
from tqdm import tqdm
import os

datadir = 'data/seir_data'
os.makedirs(datadir, exist_ok=True)

def d_by_m_e(m: List[int], e: List[int]):
    beta1, alpha, gamma_r, gamma_d1, beta2, gamma_d2 = m
    tau = 2.1
    
    def beta(t):
        return beta1 + 0.5 * np.tanh(7 * (t - tau)) * (beta2 - beta1)

    def gamma_d(t):
        return gamma_d1 + 0.5 * np.tanh(7 * (t - tau)) * (gamma_d2 - gamma_d1)

    def gamma(t):
        return gamma_r + gamma_d(t)

    def seir_model(t, y, beta, alpha, gamma):
        S, E, I, R = y
        dSdt = -beta(t) * S * I
        dEdt = beta(t) * S * I - alpha * E
        dIdt = alpha * E - gamma(t) * I
        dRdt = (gamma_r + gamma_d(t)) * I
        return [dSdt, dEdt, dIdt, dRdt]

    S0, E0, I0, R0 = 99, 1, 0, 0
    y0 = [S0, E0, I0, R0]

    solution = solve_ivp(seir_model, t_span=[0,4], y0=y0, t_eval=e, args=(beta, alpha, gamma))
    return solution.y[2:]

SIZE = 1000000

m_arr = np.zeros((SIZE, 6), dtype='float')
e_arr = np.zeros((SIZE, 4), dtype='float')
d_arr = np.zeros((SIZE, 8), dtype='float')
d_noise_arr = np.zeros((SIZE, 8), dtype='float')

for i in tqdm(range(SIZE)):
    constrain = False
    while not constrain:
        m = np.random.uniform(size=6)
        e = np.concatenate([np.random.uniform(low=1, high=1.5, size=1),
                            np.random.uniform(low=1.5, high=2, size=1),
                            np.random.uniform(low=2, high=2.5, size=1),
                            np.random.uniform(low=2.5, high=3, size=1)])
        d = d_by_m_e(m,e).flatten()
        constrain = d.shape[0] == 8
        
    m_arr[i] = m
    e_arr[i] = e
    d_arr[i] = d
    noise = np.concatenate([np.random.normal(0, 2, size=4), np.random.normal(0, 1, size=4)])
    d_noise_arr[i] = d + noise
    
np.save(f'{datadir}/m.npy', m_arr)
np.save(f'{datadir}/e.npy', e_arr)
np.save(f'{datadir}/d.npy', d_arr)
np.save(f'{datadir}/d_noise.npy', d_noise_arr)