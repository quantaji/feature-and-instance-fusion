from ..dataset import ScanNet


def get_ScanNet(dataset_dir: str, scan_id: str):
    dataset = ScanNet(dataset_root=dataset_dir)
    index = dataset.scan_id_list.index(scan_id)
    single_scan = dataset[index]

    return single_scan


def get_dataset(dataset_name: str, dataset_dir: str, scan_id: str):
    if dataset_name.lower() in ["scannet"]:
        return get_ScanNet(dataset_dir=dataset_dir, scan_id=scan_id)
    else:
        raise NotImplementedError
