conda env create -f env/conda.yaml
export CUDA_HOST_COMPILER="${HOME}/.conda/envs/feat-seg-fusion/bin/gcc"
export CUDA_PATH="${HOME}/.conda/envs/feat-seg-fusion"
export CUDA_HOME="${HOME}/.conda/envs/feat-seg-fusion"

$HOME/.conda/envs/feat-seg-fusion/bin/pip install -r env/req-basic.txt
$HOME/.conda/envs/feat-seg-fusion/bin/pip install -r env/req-captioning.txt
$HOME/.conda/envs/feat-seg-fusion/bin/pip install -r env/req-conceptfusion.txt
$HOME/.conda/envs/feat-seg-fusion/bin/pip install -r env/req-groungded-sam.txt
$HOME/.conda/envs/feat-seg-fusion/bin/pip install -r env/req-mask-rcnn.txt

# open3d for visualization
$HOME/.conda/envs/feat-seg-fusion/bin/pip install open3d

