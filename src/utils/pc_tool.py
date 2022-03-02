import numpy as np
import torch
from pyntcloud import PyntCloud
from torch import Tensor


def read_plyfile(file_name: str) -> np.array:
    """Read data from plyfile and convert numpy data

    Args:
        file_name (str): _description_
    """
    pc = PyntCloud.from_file(file_name)
    points = pc.points.to_numpy()[:, :3]
    return points


def convert_pointcloud_to_volume(points: np.array, vol_sz: int = 64) -> Tensor:
    """Sampling points into cube

    Args:
        points  (np.array): [N, 3] Point cloud data
        vol_sz (int): Volume size

    Returns:
        dense_block (Tensor): [vol_sz, vol_sz, vol_sz] _description_
    """
    v = torch.ones(points.shape[0])
    points = torch.from_numpy(points).type(torch.LongTensor)
    dense_block = torch.sparse.FloatTensor(
        torch.transpose(points, 0 , 1), v, torch.Size([vol_sz, vol_sz, vol_sz])).to_dense()

    return dense_block
