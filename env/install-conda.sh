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
# $HOME/.conda/envs/feat-seg-fusion/bin/pip install open3d
# this is the experimental version that have Visualizer.get_view_status() function implemented
$HOME/.conda/envs/feat-seg-fusion/bin/pip install https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-0.17.0+5b6ef4b-cp310-cp310-manylinux_2_27_x86_64.whl 

