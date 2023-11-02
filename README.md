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


#### sync dataset
```sh
# scene0011_00 scene0645_01 scene0643_00 scene0488_01
rsync --ignore-existing -r -v --progress -e ssh guanji@euler.ethz.ch:/cluster/project/cvg/weders/data/scannet/scans/scene0488_01 /scratch/quanta/Datasets/ScanNet/scans/

```
