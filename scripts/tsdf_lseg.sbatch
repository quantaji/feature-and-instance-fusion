#!/usr/bin/bash
#SBATCH --job-name="clip-fusion"
#SBATCH --output=%j.out
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G
#SBATCH --tmp=32G
#SBATCH --gpus=rtx_3090:2

module load gcc/8.2.0 python_gpu/3.10.4 cuda/11.8.0 git-lfs/2.3.0 git/2.31.1 eth_proxy

PYTHON=${SCRATCH}/.python_venv/feat-seg-fusion/bin/python
SCANNET_ROOT=/cluster/project/cvg/weders/data/scannet

SCENE=scene0000_00
SAVE_DIR=${SCRATCH}/Experiments/feature-instance-fusion/scannet_${SCENE}
$PYTHON fusion.py --dataset_dir $SCANNET_ROOT --scan_id $SCENE --pipeline tsdf_feature --pipeline_device cuda:1 --start 0 --end -1 --stride 1 --save_dir $SAVE_DIR --tsdf_unpruned --extractor_device cuda:0 --extractor lseg --feat_dim 512 --output_height 480 --output_width 640
