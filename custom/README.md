## The content of this directory should be placed under `<NeRF-OSR_home>/custom`.

### Files
* `env_json_to_txt.py`: NeRF-OSR provides script to extract env lighting parameters (degree 3 spherical harmonics) in a json dump. This script converts the json to txt files that is required by the dataset.
* `{run/render/test}_*.sh`: The shell script for running a training/testing job. Some require parameters like sequence id, split to render, lighting parameter location. Example: 
  ```
   sbatch   --partition=shenlong2  --time=6:00:00 --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=1 --mem=16000 ./custom/run_novel.sh 090415
  ```
* `batch_mask_pp.py`: post process for sky-masked sequences. Adds back the gt sky.  
* `mov.py`: The script for generating novel lighting by rotating the recovered parameters. It requires the starting and ending sequence id as well as the data location.  
* `animate.py`: animation script, convert images into gif.  
