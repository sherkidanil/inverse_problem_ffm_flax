import numpy as np
from tqdm import tqdm

fields = np.load('data/try1/fields_100k.npy')
solutions = np.load('data/try1/solutions_100k.npy')
features = np.zeros((solutions.shape[0]*7, solutions.shape[1], solutions.shape[2], solutions.shape[3]))
targets = np.zeros((solutions.shape[0]*7, solutions.shape[1], solutions.shape[2], solutions.shape[3]))

def create_mask(array, rows, cols, cell_size, points_count):
    mask = np.zeros_like(array)
    used_sectors = set()
    for _ in range(points_count):
        while True:
            i = np.random.randint(rows)
            j = np.random.randint(cols)
            if (i, j) not in used_sectors:
                used_sectors.add((i, j))
                break

        start_row = i * cell_size
        end_row = (i + 1) * cell_size
        start_col = j * cell_size
        end_col = (j + 1) * cell_size

        center_row = start_row + cell_size // 2 + np.random.randint(-cell_size // 7, cell_size // 7)
        center_col = start_col + cell_size // 2 + np.random.randint(-cell_size // 7, cell_size // 7)

        mask[center_row, center_col] = 1

    return mask

def create_mask_with_bc(arr, n_points):
    div = 3 + n_points // 10
    cell_size = 64 // div
    rows = div
    cols = div
    mask = create_mask(fields[0][0], rows, cols, cell_size, n_points)
    mask[ 0, : ] = 1
    mask[ -1, : ] = 1
    return mask

for i in tqdm(range(features.shape[0]//7)):
    # n_points = np.random.randint(3, 9)
    for n_points in range(3, 10):
        mask = create_mask_with_bc(solutions[i][0], n_points)
        targets[i*7 + n_points - 3][0] = fields[i][0]
        features[i*7 + n_points - 3][0] = mask * solutions[i][0]

np.save('data/try1/features_7p.npy', features)
np.save('data/try1/targets_7p.npy', targets)

