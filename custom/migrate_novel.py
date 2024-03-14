import sys
from kitti360 import KittiDataset
from highbay import HighbayDataset
import os
import numpy as np
from pathlib import Path
import imageio.v3 as iio
import json
sys.path.append("/projects/perception/personals/bohan/NeRF-OSR/colmap_runner")
from normalize_cam_dict import normalize_cam_dict


def get_tf_cams(cam_dict, target_radius=1.):
    cam_centers = []
    for im_name in cam_dict:
        W2C = np.array(cam_dict[im_name]['W2C']).reshape((4, 4))
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        center += np.array([[0], [0], [2]]) * diagonal
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    scale = target_radius / radius

    return translate, scale



def get_poses(in_cam_dict_file, source_poses, target_radius=1.):
    with open(in_cam_dict_file) as fp:
        in_cam_dict = json.load(fp)

    translate, scale = get_tf_cams(in_cam_dict, target_radius=target_radius)

    def transform_pose(W2C, translate, scale):
        C2W = np.linalg.inv(W2C)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        return np.linalg.inv(C2W)
    new_poses = []
    for P in source_poses:
        P = np.vstack([P, np.array([0,0,0,1.])])
        new_w2c = transform_pose(np.linalg.inv(P), translate, scale)
        new_poses.append(np.linalg.inv(new_w2c))
    return new_poses

    



D = KittiDataset("/projects/perception/personals/zhihao/KITTI-360", 'test', frame_start=50, frame_end=1601, test_id=[1545,1555,1565,1575,1585,1595])
root_dir = D.root_dir
imsize = [D.img_wh[0], D.img_wh[1]]
rgb_size = (imsize[1], imsize[0], 3)
mask_size = (imsize[1], imsize[0])
K = np.eye(4)
K[:3, :3] = D.K


base_dir = Path("/projects/perception/datasets/nerfosr/kitti/dev")

poses = D.render_c2w.cpu().numpy()
new_poses = get_poses(base_dir / 'kai_cameras.json', poses)

for i in range(len(new_poses)):
    fn = '{:0>6}.txt'.format(i)
    np_dir = (base_dir / 'new_poses')
    np_dir.mkdir(exist_ok=True)
    with open(np_dir / fn, 'w') as f:
        f.write(' '.join(str(k) for k in new_poses[i].flatten()))


