#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate nerfplusplus
python ddp_test_nerf.py --config configs/kitti/$1.txt --render_splits train