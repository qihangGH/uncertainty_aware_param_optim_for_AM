import argparse
import os
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_filepath', type=str, default=
r'scan_params/warp.npy'
                    )
parser.add_argument('-s', '--save_dir', type=str, default=
r'cache/warp'
                    )
parser.add_argument('--grid_size', type=float, default=0.02)
parser.add_argument('--out_stretch_size', type=float, default=0.4)
parser.add_argument('--radius', type=int, default=15)
parser.add_argument('--exp_rate', type=float, default=0.001)
parser.add_argument('--dt', type=float, default=0.00002)  # in second
parser.add_argument('--p_max', type=float, default=250)  # in Watt
args = parser.parse_args()


def get_cache(scan_info, radius, save_dir, grid_size, out_stretch_size,
              part_length, part_width, exp_rate, dt):
    if os.path.isfile(os.path.join(save_dir, f't_maps_{radius}_{exp_rate}.npy')):
        print(os.path.join(save_dir, f't_maps_{radius}_{exp_rate}.npy') + ' exists')
        return
    # scan_info [x, y, p] -> [x, y, p, v, t]
    temp_speed = \
        np.hstack([np.zeros(1), np.linalg.norm(scan_info[2:, :2] - scan_info[:-2, :2], axis=1) / (2 * dt), np.zeros(1)])
    # Note that time indices starts from 1
    scan_info = np.hstack([scan_info, temp_speed.reshape((-1, 1)), np.arange(1, len(scan_info) + 1).reshape(-1, 1)])
    num_grid_rows = int((part_width + 2 * out_stretch_size) * 1000 // (grid_size * 1000))
    num_grid_cols = int((part_length + 2 * out_stretch_size) * 1000 // (grid_size * 1000))
    coord_top_left = [np.min(scan_info[:, 0]) - out_stretch_size, np.max(scan_info[:, 1]) + out_stretch_size]
    # the grid indexed by (i, j) is
    # x∈[coord_top_left[0] + j*grid_size, coord_top_left[0] + (j+1)*grid_size)
    # y∈[coord_top_left[1] - i*grid_size, coord_top_left[1] - (i+1)*grid_size)
    j_index = ((scan_info[:, 0] - coord_top_left[0]) // grid_size).astype(int)
    i_index = ((coord_top_left[1] - scan_info[:, 1]) // grid_size).astype(int)
    # ensure that indices in a grid are in ascending order
    layer_idx_in_ij = [[[] for _ in range(num_grid_cols)] for _ in range(num_grid_rows)]
    for k, (i, j) in tqdm(enumerate(zip(i_index, j_index))):
        idx_in_ij = layer_idx_in_ij[i][j]
        # empty grid, just append
        if len(idx_in_ij) == 0:
            idx_in_ij.append(np.array([k]))
        # not empty, assert if continuous to the last sequence
        elif scan_info[idx_in_ij[-1][-1], 4] + 1 == scan_info[k, 4]:
            # enlarge the last sequence
            idx_in_ij[-1] = np.append(idx_in_ij[-1], k)
        # not empty and not continuous to the last sequence, just start a new sequence
        else:
            idx_in_ij.append(np.array([k]))
    num_max_points = 0
    for i, j in tqdm(zip(i_index, j_index)):
        if len(layer_idx_in_ij[i][j]) > 0:
            temp_max = max([len(g) for g in layer_idx_in_ij[i][j]])
            if temp_max > num_max_points:
                num_max_points = temp_max
    print(f'Maximum number of points in a grid: {num_max_points}')
    # each element is an m * 4 ndarray, where m is the number of points in a neighborhood,
    # 4 data points are (1) row index, (2) col index, (3) start, and (4) end laser time of a grid in a neighborhood
    record_p_seqs = []
    record_p_maps = [] # each element is a (2 * radius + 1) * (2 * radius + 1) ndarray
    # record_v_maps = [] # each element is a (2 * radius + 1) * (2 * radius + 1) ndarray
    record_t_maps = [] # each element is a (2 * radius + 1) * (2 * radius + 1) ndarray
    # for non-empty each grid
    # progress_bar = tqdm(range(num_grid_rows), total=num_grid_rows)  # make the loop slower
    for row in tqdm(range(num_grid_rows)):
        for col in range(num_grid_cols):
            # progress_bar.set_description(f'{row} {col}')
            for k in range(len(layer_idx_in_ij[row][col])):
                p_map = np.zeros((2 * radius + 2, 2 * radius + 2))
                # v_map = np.zeros((2 * radius + 2, 2 * radius + 2))
                t_map = np.zeros((2 * radius + 2, 2 * radius + 2))
                start_row, end_row = row - radius, row + radius + 2
                start_col, end_col = col - radius, col + radius + 2
                current_time = scan_info[layer_idx_in_ij[row][col][k][-1], 4]
                p_seq = []
                for local_i, i in enumerate(range(start_row, end_row)):
                    for local_j, j in enumerate(range(start_col, end_col)):
                        ind_in_ij = layer_idx_in_ij[i][j]
                        if len(ind_in_ij) > 0:
                            m = 0
                            nearest_m = -1
                            # find the nearest sequence to current center grid
                            # ensure that indices in a grid are in ascending order
                            while m < len(ind_in_ij) and \
                                    scan_info[ind_in_ij[m][-1], 4] <= current_time:
                                # <= should be used because the current grid must be added
                                nearest_m = m
                                m += 1
                            if nearest_m >= 0:
                                p_seq.append([local_i, local_j, ind_in_ij[nearest_m][0], ind_in_ij[nearest_m][-1]])
                                p_map[local_i, local_j] = np.mean(scan_info[ind_in_ij[nearest_m], 2])
                                t_map[local_i, local_j] = np.mean(scan_info[ind_in_ij[nearest_m], 4])
                record_p_seqs.append(np.array(p_seq, dtype=int))
                record_p_maps.append(p_map)
                record_t_maps.append(t_map)
    record_p_seqs = np.array(record_p_seqs, dtype=object)
    record_p_maps, record_t_maps = np.stack(record_p_maps), np.stack(record_t_maps)
    # obtain center p indices
    center_p_ind = []
    for seq in tqdm(record_p_seqs):
        center_p_ind.append(seq[(seq[:, 0] == radius) & (seq[:, 1] == radius), 2:])
    center_p_ind = np.squeeze(np.array(center_p_ind, dtype=int))
    # post-process t maps
    center_t = record_t_maps[:, radius, radius]
    assert np.all(center_t == np.max(record_t_maps, axis=(1, 2)))
    # TODO: Must ensure that the minimum time index is greater than 0
    zero_grids = record_t_maps == 0
    non_zero_grids = record_t_maps > 0
    record_t_maps = record_t_maps - center_t[:, None, None]
    record_t_maps[zero_grids] = 0.
    record_t_maps[non_zero_grids] = np.exp(exp_rate * record_t_maps[non_zero_grids])
    record_t_maps[non_zero_grids] = np.clip(record_t_maps[non_zero_grids], a_min=10 / 255, a_max=None)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'center_p_ind_{radius}.npy'), center_p_ind)
    np.save(os.path.join(save_dir, f'p_seqs_{radius}.npy'), record_p_seqs)
    np.save(os.path.join(save_dir, f'p_maps_{radius}.npy'), record_p_maps)
    np.save(os.path.join(save_dir, f't_maps_{radius}_{exp_rate}.npy'), record_t_maps)
    return record_p_seqs, center_p_ind, record_p_maps, record_t_maps


if __name__ == '__main__':
    scan_info = np.load(args.data_filepath)
    # Normalize the coordinates
    scan_info[:, 0] -= np.mean(scan_info[:, 0])
    scan_info[:, 1] -= np.mean(scan_info[:, 1])
    # Normalize the laser power
    scan_info[:, 2] /= args.p_max
    get_cache(scan_info, radius=args.radius, save_dir=args.save_dir, grid_size=args.grid_size,
              out_stretch_size=args.out_stretch_size,
              part_length=np.max(scan_info[:, 0]) - np.min(scan_info[:, 0]),
              part_width=np.max(scan_info[:, 1]) - np.min(scan_info[:, 1]),
              exp_rate=args.exp_rate,
              dt=args.dt)
