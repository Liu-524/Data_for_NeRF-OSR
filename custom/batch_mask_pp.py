from merge_mask import merge 
import sys
import glob
import os
import re
import cv2
from pathlib import Path
source_dir = '/projects/perception/personals/bohan/NeRF-OSR/logs/kitti_dev/render_r-30_760000/'
bg_dir = '/projects/perception/datasets/nerfosr/kitti/dev/rgb/'
dest_dir = '/projects/perception/datasets/nerfosr/kitti/dev/mask/'
out_dir = Path("movies/{}".format(source_dir.split('/') [-2]))
out_dir.mkdir(exist_ok = True)
files = os.listdir(source_dir)
out_files = [x for x in files if re.match("[0-9]*.png", x)]
mask_dir = '/projects/perception/datasets/nerfosr/kitti/dev/mask/'
start_frame = 1538 
end_frame = 1601


for i in range(end_frame - start_frame + 1):
    original_fn = '0_{:0>10d}.png'.format(i + start_frame)
    new_fn = "{:0>6d}.png".format(i)
    merge(source_dir + new_fn, 
    bg_dir + original_fn,
    mask_dir + original_fn,
    out_dir / new_fn
    )