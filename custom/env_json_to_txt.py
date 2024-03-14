#!/usr/bin/env python3
import numpy as np
import json
import sys


dest = "/projects/perception/datasets/nerfosr/kitti/090415/novel/envmaps"


with open(sys.argv[1], 'r') as f:
    envs = json.load(f)

for fn in envs:
    real_fn = fn.split('/')[-1].replace('-', '.')
    print(real_fn)
    np.savetxt(dest+ '/' + real_fn.replace('png', 'txt'), envs[fn])

        


