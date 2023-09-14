# load modules
module load gcc/8.2.0 python_gpu/3.10.4 cuda/11.8.0 git-lfs/2.3.0 git/2.31.1 eth_proxy

# create virtual env
rm -rf ${SCRATCH}/.python_venv/feat-seg-fusion
python -m venv ${SCRATCH}/.python_venv/feat-seg-fusion --upgrade-deps

# actiavte
source "${SCRATCH}/.python_venv/feat-seg-fusion/bin/activate"

export CUDA_PATH=$CUDA_HOME
export CUDA_HOST_COMPILER=$(which gcc)

# install dependency
${SCRATCH}/.python_venv/feat-seg-fusion/bin/pip3 install -r env/req-basic.txt --cache-dir ${SCRATCH}/pip_cache
${SCRATCH}/.python_venv/feat-seg-fusion/bin/pip3 install -r env/req-captioning.txt --cache-dir ${SCRATCH}/pip_cache
${SCRATCH}/.python_venv/feat-seg-fusion/bin/pip3 install -r env/req-conceptfusion.txt --cache-dir ${SCRATCH}/pip_cache
${SCRATCH}/.python_venv/feat-seg-fusion/bin/pip3 install -r env/req-groungded-sam.txt --cache-dir ${SCRATCH}/pip_cache
${SCRATCH}/.python_venv/feat-seg-fusion/bin/pip3 install -r env/req-mask-rcnn.txt --cache-dir ${SCRATCH}/pip_cache

# open3d
${SCRATCH}/.python_venv/feat-seg-fusion/bin/pip3 install "https://github.com/quantaji/open3d-manylinux2014/releases/download/0.17.0/open3d_cpu-0.17.0-0minimal-cp310-cp310-manylinux_2_17_x86_64.whl"

# deactivate
deactivate
