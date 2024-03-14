#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate nerfplusplus
python ddp_test_nerf.py --config configs/kitti/$1.txt --render_splits $2 --test_env $DATASET_DIR/nerfosr/kitti/$1/movie/$3
echo "done!"