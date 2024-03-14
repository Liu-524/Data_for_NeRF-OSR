import os
import glob
import sys
import cv2
import numpy as np
import json
from pathlib import Path
import shutil
sys.path.append("colmap_runner")
from normalize_cam_dict import normalize_cam_dict

path = sys.argv[1]
dir_name = path.split("/")[-1]

root = path[: -len(dir_name) ]
Path(root+"/temp").mkdir(parents=True, exist_ok=False)
shutil.move(path, root+"/temp")
os.rename(root+"/temp", path)
os.rename(path+"/" +dir_name, path + "/dev")
path += "/dev"
Path(path+"/mask").mkdir(parents=True, exist_ok=True)
Path(path+"/train/rgb").mkdir(parents=True, exist_ok=True)
Path(path+"/test/rgb").mkdir(parents=True, exist_ok=True)
Path(path+"/validation/rgb").mkdir(parents=True, exist_ok=True)
mask = None
intrinsics = None
data = {}

with open(path + "/intrinsics.txt") as f:
    intrinsics = [float(i) for i in f.read().split()]

for fn in glob.glob(path+"/rgb/*"):

    img = cv2.imread(fn)
    mask = np.ones(img.shape) * 255
    im_size = img.shape
    cv2.imwrite(fn.replace("rgb", "mask"), mask)
    imgfile = fn.split("/")[-1]
    with open (fn.replace("rgb", "pose").replace("png", "txt")) as f:
        w2c = [float(i) for i in f.readline().split()]
        data[imgfile] = {"K": intrinsics, "W2C": w2c, "img_size" : [im_size[1], im_size[0]]}
    split = "/train"
    if(imgfile[0] == "1"):
        split = "/test"
    else:
        if imgfile[5:7] in ["20", "40", "60", "80", "00"]:
            split = "/validation"

    shutil.move(fn, path + split + "/rgb/" + imgfile)
with open(path + '/kai_cameras.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
normalize_cam_dict(path + "/kai_cameras.json", path + "/kai_cameras_normalized.json")

cwd = os.getcwd()
os.chdir(path)
os.system("python " + cwd + "/colmap_runner/cvt.py")
os.chdir(cwd)
print("done!")
