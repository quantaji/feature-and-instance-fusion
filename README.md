# Semester Project: Fusing 2D features and segmentations into 3D

## Setup Environments


## sync command
```sh
rsync --ignore-existing -r -v --progress -e ssh guanji@euler.ethz.ch:/cluster/scratch/guanji/Experiments/feature-instance-fusion /storage/quanta/Experiments/
# if you doubt the file is corrupted use
rsync --checksum -r -v --progress -e ssh guanji@euler.ethz.ch:/cluster/scratch/guanji/Experiments/feature-instance-fusion /storage/quanta/Experiments/
# check diff
rsync -avnc   -e ssh guanji@euler.ethz.ch:/cluster/scratch/guanji/Experiments/feature-instance-fusion /storage/quanta/Experiments/
```
to sync from local to server
```
rsync --ignore-existing -r -v --progress -e ssh /storage/quanta/Experiments/feature-instance-fusion guanji@euler.ethz.ch:/cluster/scratch/guanji/Experiments/ 
```
