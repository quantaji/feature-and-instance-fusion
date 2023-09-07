conda env create -f env/conda.yaml
export CUDA_HOST_COMPILER="${HOME}/.conda/envs/feat-seg-fusion/bin/gcc"
export CUDA_PATH="${HOME}/.conda/envs/feat-seg-fusion"
export CUDA_HOME="${HOME}/.conda/envs/feat-seg-fusion"

~/.conda/envs/feat-seg-fusion/bin/pip install -r env/req-basic.txt
~/.conda/envs/feat-seg-fusion/bin/pip install -r env/req-captioning.txt
~/.conda/envs/feat-seg-fusion/bin/pip install -r env/req-conceptfusion.txt
~/.conda/envs/feat-seg-fusion/bin/pip install -r env/req-groungded-sam.txt
~/.conda/envs/feat-seg-fusion/bin/pip install -r env/req-mask-rcnn.txt

# open3d for visualization
~/.conda/envs/feat-seg-fusion/bin/pip install open3d
