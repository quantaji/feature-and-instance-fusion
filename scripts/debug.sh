export CUDA_DEVICE_ORDER=PCI_BUS_ID

# integrate tsdf
$HOME/.conda/envs/sp3/bin/python fusion.py --pipeline tsdf --pipeline_device cuda:1 --start 0 --end 100 --stride 1 --save_dir $HOME/Experiments/feature-instance-fusion/debug --tsdf_unpruned

# integrate color
$HOME/.conda/envs/sp3/bin/python fusion.py --pipeline tsdf_feature --pipeline_device cuda:1 --start 0 --end 100 --stride 1 --save_dir $HOME/Experiments/feature-instance-fusion/debug --extractor color --feat_dim 3

# integrate ground truth semantic labels
$HOME/.conda/envs/sp3/bin/python fusion.py --pipeline tsdf_feature --pipeline_device cuda:1 --start 0 --end 100 --stride 1 --save_dir $HOME/Experiments/feature-instance-fusion/debug --extractor gt_semantic --feat_dim 41 --gt_num_classes 41

# integrating lseg feature, 6.32 it /3
$HOME/.conda/envs/sp3/bin/python fusion.py --pipeline tsdf_feature --pipeline_device cuda:1 --start 0 --end 100 --stride 10 --save_dir $HOME/Experiments/feature-instance-fusion/debug --extractor lseg --feat_dim 512 --extractor_device cuda:0

# integrating conceptfusion 6.7M per frame 5.27s/it in extraction,  4.49it/s fusion
$HOME/.conda/envs/sp3/bin/python fusion.py --pipeline tsdf_feature --pipeline_device cuda:1 --start 0 --end 100 --stride 10 --save_dir $HOME/Experiments/feature-instance-fusion/debug --extractor conceptfusion --feat_dim 512 --extractor_device cuda:0 --save_extraction

# random sam: 6.4M per frame, 4.83s/it in extraction 5.97it/s in fusion
$HOME/.conda/envs/sp3/bin/python fusion.py --pipeline tsdf_feature --pipeline_device cuda:1 --start 0 --end 100 --stride 10 --save_dir $HOME/Experiments/feature-instance-fusion/debug --extractor random_sam --feat_dim 128 --extractor_device cuda:0 --save_extraction

# random grounded sam: 1.61it/s 2.2M max, 28.22it in fusion
$HOME/.conda/envs/sp3/bin/python fusion.py --pipeline tsdf_feature --pipeline_device cuda:1 --start 0 --end 100 --stride 10 --save_dir $HOME/Experiments/feature-instance-fusion/debug --extractor random_grounded_sam --feat_dim 128 --extractor_device cuda:0 --save_extraction

# conceptfusion in gradslam 2.47it/s
$HOME/.conda/envs/sp3/bin/python fusion.py --pipeline gradslam_feature --pipeline_device cuda:1 --start 0 --end 100 --stride 10 --save_dir $HOME/Experiments/feature-instance-fusion/debug --extractor conceptfusion --feature_dtype half --extractor_device cuda:0 --save_extraction --output_height 240 --output_width 320

# test panoptic fusion with ground truth, 14it/s for mesh_rc, 18it/s for sensor, 14it/s for voxel_rc, 2.51 it/s for voxel_rc_torch
$HOME/.conda/envs/sp3/bin/python fusion.py --pipeline tsdf_panoptic --pipeline_device cuda:1 --start 0 --end 100 --stride 1 --save_dir $HOME/Experiments/feature-instance-fusion/debug --extractor gt_instance --extractor_device cuda:0 --panoptic_threshold 0.25 --tsdf_depth_type mesh_rc

# test panoptic fusion with grounded sam
$HOME/.conda/envs/sp3/bin/python fusion.py --pipeline tsdf_panoptic --pipeline_device cuda:1 --start 0 --end 100 --stride 1 --save_dir $HOME/Experiments/feature-instance-fusion/debug --extractor grounded_sam --extractor_device cuda:0 --panoptic_threshold 0.25 --tsdf_depth_type mesh_rc --save_extraction

# test panoptic fusion with mask rcnn: 16.41 it/s extraction 21.53it/s fusion
$HOME/.conda/envs/sp3/bin/python fusion.py --pipeline tsdf_panoptic --pipeline_device cuda:1 --start 0 --end 100 --stride 1 --save_dir $HOME/Experiments/feature-instance-fusion/debug --extractor mask_rcnn --extractor_device cuda:0 --panoptic_threshold 0.25 --tsdf_depth_type mesh_rc --save_extraction
