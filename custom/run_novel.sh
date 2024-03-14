#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate nerfplusplus
python ddp_test_nerf.py --config configs/hibay/$1.txt --render_splits novel --test_env $DATASET_DIR/nerfosr/hibay/$1/novel/envmaps
echo "done!"