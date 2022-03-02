import glob
import os
import sys
import time

import h5py
import numpy as np
import torch.utils.data
from torch.utils.data.sampler import Sampler
from tqdm import tqdm


def read_h5file(path_file):
    pc = h5py.File(path_file, 'r')['data'][:]
    coords = pc[:, 0:3].astype('int')

    return coords


class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)


def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1
    list_data = new_list_data
    if len(list_data) == 0:
        raise ValueError('No data in the batch')
    # coords, feats = list(zip(*list_data))
    return list_data  # coords_batch, feats_batch


class PointCloudDataset(torch.utils.data.Dataset):

    def __init__(self, files):
        self.files = []
        self.cache = {}
        self.last_cache_percent = 0
        self.files = files

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):
        filedir = self.files[idx]

        if idx in self.cache:
            points = self.cache[idx]
        else:
            # points = h5py.File(filedir, 'r')['data'][:][:,0:3].astype('int')
            points = read_h5file(filedir)
            # if filedir.endswith('.h5'): coords = read_h5_geo(filedir)
            # if filedir.endswith('.ply'): coords = read_ply_ascii_geo(filedir)
            # feats = np.expand_dims(np.ones(coords.shape[0]), 1).astype('int')
            # cache
            self.cache[idx] = points
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                self.last_cache_percent = cache_percent

        # feats = feats.astype("float32")

        return points


def make_data_loader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=6,
    repeat=False,
    collate_fn=collate_pointcloud_fn
):

    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    }
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    loader = torch.utils.data.DataLoader(dataset, **args)

    return loader


# if __name__ == "__main__":
#     filedirs = sorted(glob.glob('/home/thuytt/motconmeobuon/nhn/data/training_dataset/'+'*.h5'))
#     print(len(filedirs))
#     test_dataset = PointCloudDataset(filedirs[:100])
#     test_dataloader = make_data_loader(dataset=test_dataset, batch_size=8, shuffle=True, num_workers=1, repeat=False,
#                                         collate_fn=collate_pointcloud_fn)
#     for idx, points in enumerate(tqdm(test_dataloader)):
#         if idx < 1:
#             print("="*20, "check dataset", "="*20, "\npoints:\n", points, "\n")
#             print("points[0].shape:",points[0].shape) #(10710, 3)
#             print("len(points):",len(points)) #8

#     test_iter = iter(test_dataloader)
#     print(test_iter)
#     for i in tqdm(range(1)):
#         points = test_iter.next()
#         print("="*20, "check dataset", "="*20, "\npoints:\n", points, "\n")
#         print("points[0].shape:",points[0].shape) #(5967, 3)
#         print("len(points):",len(points)) #8
