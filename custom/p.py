import json
from functools import reduce
import numpy as np
import sys
import glob
import cv2
import os

sys.path.append(os.environ['BOHAN'] + "/NeRF-OSR/colmap_runner")
from normalize_cam_dict import normalize_cam_dict

dts = sys.argv[1]
base_dir = dts + '/dev/'
new = {}
one = os.listdir(base_dir + 'train/rgb')[0]
img_size = list(reversed(cv2.imread(base_dir + 'train/rgb/' + one).shape))[1:]
with open(base_dir + 'intrinsics.txt') as f:
    K = [float(k) for k in f.read().split()]
for fn in glob.glob(base_dir + 'pose/*'):
    name = fn.split('/')[-1].replace('txt', 'png')
    curr = {}
    with open(fn) as f:
        T = np.linalg.inv(np.array([float(x) for x in f.read().split()]).reshape((4,4)))
        curr['W2C'] = [a for b in T for a in b ]
        curr['K'] = K
        curr['img_size'] =img_size
    new[name] = curr        
with open(base_dir + 'kai_cameras.json', 'w') as out:
    json.dump(new, out)

normalize_cam_dict(base_dir+ 'kai_cameras.json', base_dir+"/kai_cameras_normalized.json")

cwd = os.getcwd()
os.chdir(base_dir)
os.system("python " + cwd + "/colmap_runner/cvt.py")
os.chdir(cwd)
print("done!")
    