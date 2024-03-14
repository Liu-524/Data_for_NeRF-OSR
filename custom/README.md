# Data_scripts_NeRF-OSR
This repository holds scripts to transform data into NeRF-OSR data file structures.

## Files
`base.py` `highbay.py` `kitti360.py` `waymo.py`: These are dataloaders written by Zhi-Hao Lin. They are reused for data transform.  
`migrate_*.py`: Setting the right path to the dataset and running one of the scripts transforms the corresponding dataset. Note that `migrate_htest.py` also asks for `seq_id` (source sequence for test poses) and `ref_seq_id` (source sequence for test gt rgb).   
```
#parameters:
frame_start = 0
frame_end = 100
seq_id='waymo'
```
Also that for some datasets there are stereo image input, and it is reflected in the implementation of the `process()` function.   
`move.sh` should be run after migrate script is done. It will create train/val/test splits used by NeRF-OSR.   
NOTE: `sys.path.append("<path_to_nerfosr>/colmap_runner")` add the colmap_runner dir to path temporarily as it is needed to normalize the camera poses within unit sphere (Needed by NeRF-OSR).  
  
## Location
The scripts should be located in the dataset directory (say, kitti, with subdirectories like seq_687).   
