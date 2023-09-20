import os

import numpy as np
import torch
from scipy.sparse import csgraph, csr_matrix


from .args import ProgramArgs


def graph_connect(args: ProgramArgs):
    """
    This pipeline buid:
        border graph between diffferent patchs
        weighted connectivity graph from patch-mask correspondence
    """

    kmeans_save_dir = os.path.join(args.save_dir, "kmeans_" + args.kmeans_extractor)
    kmeans_labels: torch.Tensor = torch.load(os.path.join(kmeans_save_dir, "kmeans_labels.pt")).to(args.pipeline_device)

    graph_saving_name = args.graph_weight_method if args.graph_weight_method != "" else "binary"
    graph_save_dir = os.path.join(args.save_dir, "connectivity_graph_ext-" + args.extractor + "_kmeans-ext-" + args.kmeans_extractor + "_" + graph_saving_name)

    border_counts = np.load(os.path.join(graph_save_dir, "border_counts.npy"))
    positive_connectivity = np.load(os.path.join(graph_save_dir, "positive_connectivity.npy"))
    negative_connectivity = np.load(os.path.join(graph_save_dir, "negative_connectivity.npy"))

    log_ratio = np.log(1e-8 + positive_connectivity) - np.log(1e-8 + negative_connectivity)

    adjacency_matrix = (positive_connectivity > args.positive_threshold) * (log_ratio > args.log_ratio_threshold) * (border_counts > 0)

    sparse_matrix = csr_matrix(adjacency_matrix)
    _, labels = csgraph.connected_components(sparse_matrix)

    voxel_labels = torch.from_numpy(labels).to(device=args.pipeline_device, dtype=torch.int64)[kmeans_labels]
    # also add a unique to reduce total amount
    voxel_labels = voxel_labels.unique(sorted=True, return_inverse=True)[1].detach().cpu()

    save_dir = os.path.join(args.save_dir, "graph_connect_etx-" + args.extractor + "_kmeans-ext-" + args.kmeans_extractor + "_" + graph_saving_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(obj=voxel_labels, f=os.path.join(save_dir, "merged_labels.pt"))
