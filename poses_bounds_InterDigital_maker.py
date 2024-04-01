import numpy as np
# import pandas as pd
from pathlib import Path
import os

dataset_dirpath = Path('data/InterDigital/')
scenes = os.listdir(dataset_dirpath)
scenes = [scene for scene in scenes if scene != 'train_test_sets']

for scene in scenes:
    camera_intrinsics_path = f'{dataset_dirpath}/{scene}/CameraIntrinsics_undistorted.csv'
    camera_extrinsics_path = f'{dataset_dirpath}/{scene}/CameraExtrinsics_undistorted.csv'
    camera_depth_bounds_path = f'{dataset_dirpath}/{scene}/DepthBounds.csv'

    camera_intrinsics = np.loadtxt(camera_intrinsics_path, delimiter=',')
    camera_extrinsics = np.loadtxt(camera_extrinsics_path, delimiter=',')
    camera_depth_bounds = np.loadtxt(camera_depth_bounds_path, delimiter=',')

    camera_intrinsics = camera_intrinsics.reshape(-1, 3, 3)
    camera_extrinsics = camera_extrinsics.reshape(-1, 4, 4)
    camera_depth_bounds = camera_depth_bounds.reshape(-1, 2)

    h = camera_intrinsics[0, 1, 2]
    w = camera_intrinsics[0, 0, 2]
    f = camera_intrinsics[0, 0, 0]
    hwf = np.array([h, w, f])
    I = np.repeat(hwf[np.newaxis, :], camera_intrinsics.shape[0], axis=0)

    E = camera_extrinsics
    E_inv = np.linalg.inv(E)
    permuter = np.eye(4)
    permuter[:2] = permuter[:2][::-1]
    permuter[2,2] *= -1
    E = permuter @ E @ np.linalg.inv(permuter)
    E = E[:, :3, :4]
    E.reshape(-1, 12)

    poses_bounds = np.concatenate([E.reshape(-1, 12), I, camera_depth_bounds], axis=1)
    np.save(f'{dataset_dirpath}/{scene}/poses_bounds.npy', poses_bounds)



