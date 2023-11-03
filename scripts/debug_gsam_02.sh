# grounede sam, 1.18 it/s extraction, fusion 20 it/s
$HOME/.conda/envs/feat-seg-fusion/bin/python fusion.py --pipeline tsdf_feature --pipeline_device cuda:0 --start 0 --end -1 --stride 1 --save_dir /scratch/quanta/Experiments/feature-instance-fusion-02/scannet_scene0000_00 --extractor random_grounded_sam --feat_dim 128 --extractor_device cuda:0 --save_extraction --dataset_dir /scratch/quanta/Datasets/ScanNet --grounded_sam_tag_set built_in

# kmeans
$HOME/.conda/envs/feat-seg-fusion/bin/python fusion.py --pipeline kmeans --pipeline_device cuda:0 --save_dir /scratch/quanta/Experiments/feature-instance-fusion-02/scannet_scene0000_00 --kmeans_cluster_num 1024 --kmeans_extractor random_grounded_sam --kmeans_position_factor 0.02

# outlier removal
$HOME/.conda/envs/feat-seg-fusion/bin/python fusion.py --pipeline outlier_removal --pipeline_device cuda:0 --save_dir /scratch/quanta/Experiments/feature-instance-fusion-02/scannet_scene0000_00 --kmeans_extractor random_grounded_sam --label_outlier_removal_nb_neighbors 20 --label_outlier_removal_std_ratio 2.0

# patch_corres 3.5it /s
$HOME/.conda/envs/feat-seg-fusion/bin/python fusion.py --pipeline patch_corres --pipeline_device cuda:0 --start 0 --end -1 --stride 1 --save_dir /scratch/quanta/Experiments/feature-instance-fusion-02/scannet_scene0000_00 --kmeans_cluster_num 1024 --kmeans_extractor random_grounded_sam --extractor grounded_sam --dataset_dir /scratch/quanta/Datasets/ScanNet 

# build graph 112it/s
$HOME/.conda/envs/feat-seg-fusion/bin/python fusion.py --pipeline build_graph --pipeline_device cuda:0 --save_dir /scratch/quanta/Experiments/feature-instance-fusion-02/scannet_scene0000_00 --kmeans_cluster_num 1024 --kmeans_extractor random_grounded_sam --extractor grounded_sam --dataset_dir /scratch/quanta/Datasets/ScanNet

# graph connect
$HOME/.conda/envs/feat-seg-fusion/bin/python fusion.py --pipeline graph_connect --pipeline_device cuda:0 --save_dir /scratch/quanta/Experiments/feature-instance-fusion-02/scannet_scene0000_00 --kmeans_cluster_num 1024 --kmeans_extractor random_grounded_sam --extractor grounded_sam --dataset_dir /scratch/quanta/Datasets/ScanNet --log_ratio_threshold 1.5 --positive_threshold 2.0

# guided panoptic 15it/s
$HOME/.conda/envs/feat-seg-fusion/bin/python fusion.py --pipeline tsdf_guided_panoptic --pipeline_device cuda:0 --save_dir /scratch/quanta/Experiments/feature-instance-fusion-02/scannet_scene0000_00 --kmeans_cluster_num 1024 --kmeans_extractor random_grounded_sam --extractor grounded_sam --save_extraction --extractor_device cuda:1 --tsdf_depth_type mesh_rc  --dataset_dir /scratch/quanta/Datasets/ScanNet --tsdf_unpruned

# label extend
$HOME/.conda/envs/feat-seg-fusion/bin/python fusion.py --pipeline tsdf_label_extend --pipeline_device cuda:0 --save_dir /scratch/quanta/Experiments/feature-instance-fusion-02/scannet_scene0000_00 --kmeans_cluster_num 1024 --kmeans_extractor random_grounded_sam --extractor grounded_sam --save_extraction --extractor_device cuda:1 --tsdf_depth_type mesh_rc  --dataset_dir /scratch/quanta/Datasets/ScanNet --tsdf_unpruned

# panoptic 15 it /s
$HOME/.conda/envs/feat-seg-fusion/bin/python fusion.py --pipeline tsdf_panoptic --pipeline_device cuda:0 --start 0 --end -1 --stride 1 --save_dir /scratch/quanta/Experiments/feature-instance-fusion-02/scannet_scene0000_00 --extractor grounded_sam --save_extraction --extractor_device cuda:1 --panoptic_threshold 0.15 --tsdf_depth_type mesh_rc --dataset_dir /scratch/quanta/Datasets/ScanNet --tsdf_unpruned
