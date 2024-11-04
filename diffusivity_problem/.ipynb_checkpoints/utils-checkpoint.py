import numpy as np

def get_d_from_u(arr: np.array, points: dict):
    n = len(points)
    results = np.zeros(n)
    for i in range(n):
        x, y = points[i]
        results[i] = arr[x, y]
    return results
