import sys
from highbay import HighbayDataset
import os
import numpy as np
from pathlib import Path
import imageio.v3 as iio
import json
sys.path.append("/projects/perception/personals/bohan/NeRF-OSR/colmap_runner")
from normalize_cam_dict import normalize_cam_dict

frame_start = 50
frame_end = 390

seq_id='090415'
seq_root = str(seq_id)
rgb_out = Path(seq_root + '/rgb')
rgb_out.mkdir(exist_ok = True, parents=True)
mask_out = Path(seq_root + '/mask')
mask_out.mkdir(exist_ok = True, parents=True)

kwargs = {
    
}
D = HighbayDataset(
        f'/projects/perception/datasets/highbay/processed/corner0_{seq_id}',
        'train', nvs=False,
        load_2d=False,
        downsample=0.5,
        ** kwargs
    )

root_dir = D.root_dir
imsize = [D.img_wh[0], D.img_wh[1]]
rgb_size = (imsize[1], imsize[0], 3)
mask_size = (imsize[1], imsize[0])
K = np.eye(4)
K[:3, :3] = D.K

dir_rgb_0 = os.path.join(root_dir, 'left', 'rgb')
dir_rgb_1 = os.path.join(root_dir, 'right', 'rgb')
dir_sem_0 = os.path.join(root_dir, 'left', 'semantic')
dir_sem_1 = os.path.join(root_dir, 'right', 'semantic')
dir_normal_0 = os.path.join(root_dir, 'left', 'normal')
dir_normal_1 = os.path.join(root_dir, 'right', 'normal')
dir_shadow_0 = os.path.join(root_dir, 'left', 'shadow')
dir_shadow_1 = os.path.join(root_dir, 'right', 'shadow')
dir_deshadow_0 = os.path.join(root_dir, 'left', 'deshadow')
dir_deshadow_1 = os.path.join(root_dir, 'right', 'deshadow')




poses = D.poses.numpy()
size = len(poses) // 2
print(size)
J = {}

def process(frame_id):
    fn =  'l_{:0>5d}.png'.format(frame_id)
    rgb = (D.read_rgb(dir_rgb_0, [frame_id])[0] * 255).astype(np.uint8)
    mask = D.read_semantics(dir_sem_0, [frame_id])[0]
    mask = np.logical_and(mask != 5, mask != 5)
    
    pose = poses[frame_id]
    pose = np.vstack([pose, np.array([0,0,0,1.])])
    pose = np.linalg.inv(pose)

    iio.imwrite(rgb_out / fn, rgb.reshape(rgb_size))
    iio.imwrite(mask_out / fn, mask.reshape( mask_size ))

    frame = {"K" : [i for i in K.flatten()], 'img_size': imsize, "W2C": [i for i in pose.flatten()]}
    J[fn] = frame

    fn =  'r_{:0>5d}.png'.format(frame_id)
    rgb = (D.read_rgb(dir_rgb_1, [frame_id])[0] * 255).astype(np.uint8)
    mask = D.read_semantics(dir_sem_1, [frame_id])[0]
    mask = np.logical_and(mask != 5, mask != 5)
    pose = poses[frame_id + size]
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