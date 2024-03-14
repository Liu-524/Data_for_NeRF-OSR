import sys
import numpy as np 
from demo_projSH_rotSH import Rotation
import json
import shutil
from pathlib import Path

split_name = "movie"
base_dir = Path('/projects/perception/datasets/nerfosr/kitti/090415')
movie_dir = base_dir / split_name
movie_dir.mkdir(exist_ok=True)
# JSON_PATH = '/projects/perception/personals/bohan/NeRF-OSR/logs/kitti_dev/model_760000.pth.env_params.json'
JSON_PATH = "/projects/perception/personals/bohan/NeRF-OSR/logs/corner0_090409_2/model_050000.pth.env_params.json"
env_json = json.load(open(JSON_PATH))
start_frame = 4474
end_frame = 4537

def deg_to_rad(deg):
    return deg / 180 * np.pi


def get_env(d, i):
    return d['train/rgb/0_{:0>10d}-png'.format(start_frame + i)]

def dump_env(p, e):
    with open(p,'w') as f:
        for row in e:
            f.write(' '.join([str(x) for x in row]) + '\n')

R_15 = Rotation().rot_z(-15)
R_30 = Rotation().rot_z(-30)

env_dir_0 = movie_dir / 'envmaps-0'
intrinsics_dir = movie_dir / 'intrinsics'
pose_dir = movie_dir / 'pose'
rgb_dir = movie_dir / 'rgb'
env_dir_0.mkdir(exist_ok=True)
intrinsics_dir.mkdir(exist_ok=True)
pose_dir.mkdir(exist_ok=True)
rgb_dir.mkdir(exist_ok=True)



for i in range(end_frame - start_frame + 1):
    print(i + start_frame)
    env = np.array(get_env(env_json, i))
    dump_env(env_dir_0 / '{:0>5d}.txt'.format(i), env)
    shutil.copy(base_dir / 'train/rgb/l_{:0>5d}.png'.format(i + start_frame), rgb_dir / "{:0>5d}.png".format(i))
    shutil.copy(base_dir / 'train/intrinsics/l_{:0>5d}.txt'.format(i + start_frame), intrinsics_dir / "{:0>6d}.txt".format(i))
    shutil.copy(base_dir / 'train/pose/l_{:0>5d}.txt'.format(i + start_frame), pose_dir / "{:0>5d}.txt".format(i))






    
    
    



