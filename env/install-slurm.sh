# load modules
module load gcc/8.2.0 python_gpu/3.10.4 cuda/11.8.0 git-lfs/2.3.0 git/2.31.1 eth_proxy

# create virtual env
python -m venv ${SCRATCH}/.python_venv/feat-seg-fusion --upgrade-deps

# actiavte
source "${SCRATCH}/.python_venv/feat-seg-fusion/bin/activate"

# install dependency
${SCRATCH}/.python_venv/feat-seg-fusion/bin/pip3 install -r env/req-basic.txt --cache-dir ${SCRATCH}/pip_cache
${SCRATCH}/.python_venv/feat-seg-fusion/bin/pip3 install -r env/req-captioning.txt --cache-dir ${SCRATCH}/pip_cache
${SCRATCH}/.python_venv/feat-seg-fusion/bin/pip3 install -r env/req-conceptfusion.txt --cache-dir ${SCRATCH}/pip_cache
${SCRATCH}/.python_venv/feat-seg-fusion/bin/pip3 install -r env/req-groungded-sam.txt --cache-dir ${SCRATCH}/pip_cache
${SCRATCH}/.python_venv/feat-seg-fusion/bin/pip3 install -r env/req-mask-rcnn.txt --cache-dir ${SCRATCH}/pip_cache

# deactivate
deactivate
