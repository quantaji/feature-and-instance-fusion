#!/usr/bin/bash
#SBATCH --job-name="clip-fusion"
#SBATCH --output=%j.out
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=32G
#SBATCH --gpus=rtx_3090:1

module load gcc/8.2.0 python_gpu/3.10.4 cuda/11.8.0 git-lfs/2.3.0 git/2.31.1 eth_proxy

PYTHON=${SCRATCH}/.python_venv/feat-seg-fusion/bin/python
SCANNET_ROOT=/cluster/project/cvg/weders/data/scannet

SCENE=scene0000_00
SAVE_DIR=${SCRATCH}/Experiments/feature-instance-fusion/scannet_${SCENE}
$PYTHON fusion.py --pipeline tsdf --pipeline_device cuda:0 --start 0 --end -1 --stride 1 --save_dir SAVE_DIR --tsdf_unpruned


SCENES=( scene0011_00 scene0645_01 scene0608_00 scene0643_00 scene0488_01)
for SCENE in "${SCENES[@]}"; do
    SAVE_DIR=${SCRATCH}/Experiments/feature-instance-fusion/scannet_${SCENE}
    $PYTHON fusion.py --pipeline tsdf --pipeline_device cuda:0 --start 0 --end -1 --stride 1 --save_dir SAVE_DIR
done
