import sys
from kitti360 import KittiDataset
import os
import numpy as np
from pathlib import Path
import imageio.v3 as iio
import json
sys.path.append("/projects/perception/personals/bohan/NeRF-OSR/colmap_runner")
from normalize_cam_dict import normalize_cam_dict


frame_start = int(sys.argv[1])
frame_end = int(sys.argv[2])
seq_root = str(frame_start)
seq_id=0
rgb_out = Path(seq_root + '/rgb')
rgb_out.mkdir(exist_ok = True, parents=True)
mask_out = Path(seq_root + '/mask')
mask_out.mkdir(exist_ok = True, parents=True)

D = KittiDataset("/projects/perception/personals/zhihao/KITTI-360", 'train', frame_start=frame_start, frame_end=frame_end, test_id=[1540])
root_dir = D.root_dir
imsize = [D.img_wh[0], D.img_wh[1]]
rgb_size = (imsize[1], imsize[0], 3)
mask_size = (imsize[1], imsize[0])
K = np.eye(4)
K[:3, :3] = D.K

dir_seq = '2013_05_28_drive_{:0>4d}_sync'.format(seq_id)
dir_rgb_0 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_00', 'data_rect')
dir_rgb_1 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_01', 'data_rect')
dir_sem_0 = os.path.join(root_dir, 'data_2d_semantics/train', dir_seq, 'image_00/semantic')
dir_sem_1 = os.path.join(root_dir, 'data_2d_semantics/train', dir_seq, 'image_01/semantic')
dir_normal_0 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_00', 'normal')
dir_normal_1 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_01', 'normal')
dir_shadow_0 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_00', 'shadow')
dir_shadow_1 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_01', 'shadow')
dir_deshadow_0 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_00', 'deshadow')
dir_deshadow_1 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_01', 'deshadow')
dir_calib = os.path.join(root_dir, 'calibration')
dir_poses = os.path.join(root_dir, 'data_poses', dir_seq)
dir_oxts = os.path.join(root_dir, 'data_poses', dir_seq, 'oxts')




poses = D.poses.numpy()
J = {}

def process(frame_id):
    fn =  '0_{:0>10d}.png'.format(frame_id)
    rgb = (D.read_rgb(dir_rgb_0, [frame_id])[0] * 255).astype(np.uint8)
    mask = D.read_semantics(dir_sem_0, [frame_id])[0]
    mask = np.logical_and(mask != 5, mask != 5)
    
    pose = poses[frame_id - frame_start]
    pose = np.vstack([pose, np.array([0,0,0,1.])])
    pose = np.linalg.inv(pose)

    iio.imwrite(rgb_out / fn, rgb.reshape(rgb_size))
    iio.imwrite(mask_out / fn, mask.reshape( mask_size ))

    frame = {"K" : [i for i in K.flatten()], 'img_size': imsize, "W2C": [i for i in pose.flatten()]}
    J[fn] = frame

    fn =  '1_{:0>10d}.png'.format(frame_id)
    rgb = (D.read_rgb(dir_rgb_1, [frame_id])[0] * 255).astype(np.uint8)
    mask = D.read_semantics(dir_sem_1, [frame_id])[0]
    mask = np.logical_and(mask != 5, mask != 5)
    pose = poses[frame_id - frame_start + frame_end - frame_start + 1]
    pose = np.vstack([pose, np.array([0,0,0,1.])])
    pose = np.linalg.inv(pose)

    iio.imwrite(rgb_out / fn, rgb.reshape( rgb_size ))
    iio.imwrite(mask_out / fn, mask.reshape( mask_size ))
    frame = {"K" : [i for i in K.flatten()], 'img_size': imsize, "W2C": [i for i in pose.flatten()]}
    J[fn] = frame

for i in range(frame_start, frame_end+1):
    process(i)

json.dump(J, open(seq_root + '/kai_cameras.json', 'w'))


normalize_cam_dict(seq_root + "/kai_cameras.json", seq_root + "/kai_cameras_normalized.json")