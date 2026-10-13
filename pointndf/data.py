import os
import numpy as np
import random
import torch
import torch.utils.data.dataset
from pytorch3d.ops import knn_points
from tqdm import tqdm
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, grouping_operation

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_knn(nsample, xyz, new_xyz, include_self=True):
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)
    sorted_dist, indices = torch.sort(sqrdists, dim=-1, descending=False)
    idx = indices[:, :, pad:nsample + pad]
    sdist = sorted_dist[:, :, pad:nsample + pad]
    return (idx.int(), sdist)

def split_patches(noise_xyz, n_patches=64, nsample=1024):
    noise_xyz = noise_xyz.unsqueeze(0).cuda()
    noise_xyz_flipped = noise_xyz.permute(0, 2, 1).contiguous()
    new_xyz = gather_operation(noise_xyz_flipped, furthest_point_sample(noise_xyz, n_patches))
    idx, sqdists = query_knn(nsample, noise_xyz, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(noise_xyz_flipped, idx)
    grouped_xyz = grouped_xyz.permute(0, 2, 3, 1).contiguous()
    grouped_xyz = grouped_xyz.squeeze()
    return grouped_xyz.cpu()

def split_patches_for_pcl_pair(xyz, noise_xyz, n_patches=30, nsample=1024):
    xyz = xyz.unsqueeze(0).cuda()
    noise_xyz = noise_xyz.unsqueeze(0).cuda()
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()
    noise_xyz_flipped = noise_xyz.permute(0, 2, 1).contiguous()
    new_xyz = gather_operation(xyz_flipped, furthest_point_sample(xyz, n_patches))
    idx, sqdists = query_knn(nsample, xyz, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz_flipped, idx)
    grouped_xyz = grouped_xyz.permute(0, 2, 3, 1).contiguous()
    grouped_xyz = grouped_xyz.squeeze()
    grouped_noise_xyz = grouping_operation(noise_xyz_flipped, idx)
    grouped_noise_xyz = grouped_noise_xyz.permute(0, 2, 3, 1).contiguous()
    grouped_noise_xyz = grouped_noise_xyz.squeeze()
    return (grouped_xyz.cpu(), grouped_noise_xyz.cpu())

def make_patches_for_pcl_pair(pcl_A, pcl_B, patch_size, num_patches, ratio=1):
    N = pcl_A.size(0)
    seed_idx = torch.randperm(N)[:num_patches]
    seed_pnts = pcl_A[seed_idx].unsqueeze(0)
    _, _, pat_A = knn_points(seed_pnts, pcl_A.unsqueeze(0), K=patch_size, return_nn=True)
    pat_A = pat_A[0]
    _, _, pat_B = knn_points(seed_pnts, pcl_B.unsqueeze(0), K=int(ratio * patch_size), return_nn=True)
    pat_B = pat_B[0]
    return (pat_A, pat_B)

class ModelNetDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, root, split, transform=None):
        super().__init__()
        self.transform = transform
        self.pointclouds = []
        self.pointcloud_names = []
        catelist = os.listdir(root)
        for cate in catelist:
            pcl_dir = os.path.join(root, cate, split)
            for fn in tqdm(os.listdir(pcl_dir), desc='Loading ' + split):
                if fn[-3:] != 'xyz':
                    continue
                pcl_path = os.path.join(pcl_dir, fn)
                if not os.path.exists(pcl_path):
                    raise FileNotFoundError('File not found: %s' % pcl_path)
                pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
                self.pointclouds.append(pcl)
                self.pointcloud_names.append(fn[:-4])

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {'pcl_clean': self.pointclouds[idx].clone(), 'name': self.pointcloud_names[idx]}
        if self.transform is not None:
            data = self.transform(data)
        return data

class PUNetDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, root, split, resolution='10000_poisson', transform=None):
        super().__init__()
        self.pcl_dir = os.path.join(root, 'pointclouds', split, resolution)
        self.transform = transform
        self.pointclouds = []
        self.pointcloud_names = []
        for fn in tqdm(os.listdir(self.pcl_dir), desc='Loading'):
            if fn[-3:] != 'xyz':
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            if not os.path.exists(pcl_path):
                raise FileNotFoundError('File not found: %s' % pcl_path)
            pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
            self.pointclouds.append(pcl)
            self.pointcloud_names.append(fn[:-4])

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {'pcl_clean': self.pointclouds[idx].clone(), 'name': self.pointcloud_names[idx]}
        if self.transform is not None:
            data = self.transform(data)
        return data

class PUNetTestDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, root, resolution='10000_poisson', level=0.01):
        super().__init__()
        num = resolution.split('_')[0]
        self.pcl_dir = os.path.join(root, 'pointclouds/test/%s', resolution)
        self.pcl_clean_dir = os.path.join(root, 'pointclouds/test/%s', num + '_poisson')
        self.clean_pointclouds = []
        self.noise_pointclouds = []
        self.pointcloud_names = []
        for fn in tqdm(os.listdir(self.pcl_clean_dir % 'clean'), desc='Loading'):
            if fn[-3:] != 'xyz':
                continue
            pcl_path = os.path.join(self.pcl_clean_dir % 'clean', fn)
            if not os.path.exists(pcl_path):
                raise FileNotFoundError('File not found: %s' % pcl_path)
            pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
            self.clean_pointclouds.append(pcl)
            self.pointcloud_names.append(fn[:-4])
            pcl_noise_path = os.path.join(self.pcl_dir % 'noise' + '_' + str(level), fn)
            if not os.path.exists(pcl_noise_path):
                raise FileNotFoundError('File not found: %s' % pcl_noise_path)
            pcl_noise = torch.FloatTensor(np.loadtxt(pcl_noise_path, dtype=np.float32))
            self.noise_pointclouds.append(pcl_noise)

    def __len__(self):
        return len(self.clean_pointclouds)

    def __getitem__(self, idx):
        data = {'pcl_clean': self.clean_pointclouds[idx].clone(), 'pcl_noisy': self.noise_pointclouds[idx].clone(), 'name': self.pointcloud_names[idx]}
        return data

class PairedPatchDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, datasets, patch_size=1000, num_patches=1000, transform=None, patch_ratio=1, on_the_fly=True):
        super().__init__()
        self.datasets = datasets
        self.len_datasets = sum([len(dset) for dset in datasets])
        self.patch_ratio = patch_ratio
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.on_the_fly = on_the_fly
        self.transform = transform
        self.patches = []
        if not on_the_fly:
            self.make_patches()

    def make_patches(self):
        for dataset in tqdm(self.datasets, desc='MakePatch'):
            for data in tqdm(dataset):
                pat_noisy, pat_clean = make_patches_for_pcl_pair(data['pcl_noisy'], data['pcl_clean'], patch_size=self.patch_size, num_patches=self.num_patches, ratio=self.patch_ratio)
                for i in range(pat_noisy.size(0)):
                    self.patches.append((pat_noisy[i], pat_clean[i]))

    def __len__(self):
        if not self.on_the_fly:
            return len(self.patches)
        else:
            return self.len_datasets * self.num_patches

    def __getitem__(self, idx):
        if self.on_the_fly:
            pcl_dset = random.choice(self.datasets)
            pcl_data = pcl_dset[idx % len(pcl_dset)]
            pat_noisy, pat_clean = make_patches_for_pcl_pair(pcl_data['pcl_noisy'], pcl_data['pcl_clean'], patch_size=self.patch_size, num_patches=1, ratio=self.patch_ratio)
            data = {'pcl_noisy': pat_noisy[0], 'pcl_clean': pat_clean[0]}
        else:
            data = {'pcl_noisy': self.patches[idx][0].clone(), 'pcl_clean': self.patches[idx][1].clone()}
        return data

class PPatchDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, datasets, patch_ratio=1.0, on_the_fly=True, patch_size=1000, num_patches=1000, transform=None):
        super().__init__()
        self.datasets = datasets
        self.len_datasets = sum([len(dset) for dset in datasets])
        self.patch_ratio = patch_ratio
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.on_the_fly = on_the_fly
        self.transform = transform
        self.patches = []
        if not on_the_fly:
            self.make_patches()

    def make_patches(self):
        for dataset in tqdm(self.datasets, desc='MakePatch'):
            for data in tqdm(dataset):
                pat_noisy, pat_clean = make_patches_for_pcl_pair(data['pcl_noisy'], data['pcl_clean'], patch_size=self.patch_size, num_patches=self.num_patches, ratio=self.patch_ratio)
                for i in range(pat_noisy.size(0)):
                    self.patches.append((pat_noisy[i], pat_clean[i]))

    def __len__(self):
        if not self.on_the_fly:
            return len(self.patches)
        else:
            return self.len_datasets * self.num_patches

    def __getitem__(self, idx):
        if self.on_the_fly:
            pcl_dset = random.choice(self.datasets)
            pcl_data = pcl_dset[idx % len(pcl_dset)]
            data = {'pcl_clean': make_patches_for_pcl(pcl_data['pcl_clean'], patch_size=self.patch_size, num_patches=1)[0]}
            if self.transform is not None:
                data = self.transform(data)
                del data['noise_std']
        else:
            data = {'pcl_noisy': self.patches[idx][0].clone(), 'pcl_clean': self.patches[idx][1].clone()}
        return data
