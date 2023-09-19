# Semester Project: Fusing 2D features and segmentations into 3D

## Setup Environments


## sync command
```sh
rsync --ignore-existing -r -v --progress -e ssh guanji@euler.ethz.ch:/cluster/scratch/guanji/Experiments/feature-instance-fusion /home/quanta/Experiments/
# if you doubt the file is corrupted use
rsync --checksum -r -v --progress -e ssh guanji@euler.ethz.ch:/cluster/scratch/guanji/Experiments/feature-instance-fusion /home/quanta/Experiments/
# check diff
rsync -avnc   -e ssh guanji@euler.ethz.ch:/cluster/scratch/guanji/Experiments/feature-instance-fusion /home/quanta/Experiments/
```
to sync from local to server
```sh
rsync --ignore-existing -r -v --progress -e ssh /home/quanta/Experiments/feature-instance-fusion guanji@euler.ethz.ch:/cluster/scratch/guanji/Experiments/ 
```
