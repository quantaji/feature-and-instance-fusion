# grounede sam, 1.18 it/s extraction, fusion 20 it/s
$HOME/.conda/envs/feat-seg-fusion/bin/python fusion.py --pipeline tsdf_feature --pipeline_device cuda:0 --start 0 --end -1 --stride 1 --save_dir /scratch/quanta/Experiments/feature-instance-fusion-new/scannet_scene0000_00 --extractor random_grounded_sam --feat_dim 128 --extractor_device cuda:0 --save_extraction --dataset_dir /scratch/quanta/Datasets/ScanNet --grounded_sam_tag_set scannet_200

# kmeans
$HOME/.conda/envs/feat-seg-fusion/bin/python fusion.py --pipeline kmeans --pipeline_device cuda:0 --save_dir /scratch/quanta/Experiments/feature-instance-fusion-new/scannet_scene0000_00 --kmeans_cluster_num 1024 --kmeans_extractor random_grounded_sam --kmeans_position_factor 0.03

# outlier removal
$HOME/.conda/envs/feat-seg-fusion/bin/python fusion.py --pipeline outlier_removal --pipeline_device cuda:0 --save_dir /scratch/quanta/Experiments/feature-instance-fusion-new/scannet_scene0000_00 --kmeans_extractor random_grounded_sam --label_outlier_removal_nb_neighbors 20 --label_outlier_removal_std_ratio 2.0

# patch_corres 3.5it /s
$HOME/.conda/envs/feat-seg-fusion/bin/python fusion.py --pipeline patch_corres --pipeline_device cuda:0 --start 0 --end -1 --stride 1 --save_dir /scratch/quanta/Experiments/feature-instance-fusion-new/scannet_scene0000_00 --kmeans_cluster_num 1024 --kmeans_extractor random_grounded_sam --extractor grounded_sam --dataset_dir /scratch/quanta/Datasets/ScanNet 

# build graph 112it/s
$HOME/.conda/envs/feat-seg-fusion/bin/python fusion.py --pipeline build_graph --pipeline_device cuda:0 --save_dir /scratch/quanta/Experiments/feature-instance-fusion-new/scannet_scene0000_00 --kmeans_cluster_num 1024 --kmeans_extractor random_grounded_sam --extractor grounded_sam --dataset_dir /scratch/quanta/Datasets/ScanNet

# graph connect
$HOME/.conda/envs/feat-seg-fusion/bin/python fusion.py --pipeline graph_connect --pipeline_device cuda:0 --save_dir /scratch/quanta/Experiments/feature-instance-fusion-new/scannet_scene0000_00 --kmeans_cluster_num 1024 --kmeans_extractor random_grounded_sam --extractor grounded_sam --dataset_dir /scratch/quanta/Datasets/ScanNet --log_ratio_threshold 1.9 --positive_threshold 1.0 # 1.9 is the minimum value that I can get the bycicle to retain
