#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate nerfplusplus
python ddp_train_nerf.py --config configs/train/dev.txt