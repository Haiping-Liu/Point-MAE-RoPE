'''
@author: Modified for HDF5 format
@file: ModelNetHDF5.py
'''
import os
import numpy as np
import warnings
import h5py
import pickle
from tqdm import tqdm

from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *
import torch

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


@DATASETS.register_module()
class ModelNetHDF5(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.get('USE_NORMALS', False)
        self.num_category = config.get('NUM_CATEGORY', 40)
        self.process_data = True
        self.uniform = True
        split = config.subset
        self.subset = config.subset
        
        # Load shape names for class mapping
        self.catfile = os.path.join(self.root, 'shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        
        # Load file lists
        if self.subset == 'train':
            with open(os.path.join(self.root, 'train_files.txt'), 'r') as f:
                self.file_list = [os.path.basename(line.strip()) for line in f]
        else:
            with open(os.path.join(self.root, 'test_files.txt'), 'r') as f:
                self.file_list = [os.path.basename(line.strip()) for line in f]
        
        # Create save path for processed data
        if self.uniform:
            self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps_hdf5.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_hdf5.dat' % (self.num_category, split, self.npoints))
        
        if self.process_data:
            if not os.path.exists(self.save_path):
                print_log('Processing data %s (only running in the first time)...' % self.save_path, logger = 'ModelNetHDF5')
                
                # Load all data from HDF5 files
                all_data = []
                all_label = []
                all_normal = []
                
                for h5_filename in self.file_list:
                    f = h5py.File(os.path.join(self.root, h5_filename), 'r')
                    data = f['data'][:]
                    label = f['label'][:]
                    normal = f['normal'][:] if 'normal' in f else None
                    f.close()
                    all_data.append(data)
                    all_label.append(label)
                    if normal is not None:
                        all_normal.append(normal)
                
                self.data = np.concatenate(all_data, axis=0)
                self.label = np.concatenate(all_label, axis=0).squeeze().astype(np.int64)
                if all_normal:
                    self.normal = np.concatenate(all_normal, axis=0)
                else:
                    self.normal = None
                
                print_log('The size of %s data is %d' % (split, self.data.shape[0]), logger = 'ModelNetHDF5')
                
                # Process data
                self.list_of_points = [None] * self.data.shape[0]
                self.list_of_labels = [None] * self.data.shape[0]
                
                for index in tqdm(range(self.data.shape[0]), total=self.data.shape[0]):
                    points = self.data[index]
                    if self.normal is not None and self.use_normals:
                        point_set = np.concatenate([points, self.normal[index]], axis=1)
                    else:
                        point_set = points
                    
                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]
                    
                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = np.array([self.label[index]]).astype(np.int32)
                
                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print_log('Load processed data from %s...' % self.save_path, logger = 'ModelNetHDF5')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)
        
        # Set datapath for compatibility
        self.datapath = [(None, None)] * len(self.list_of_points)

    def __len__(self):
        return len(self.list_of_points)

    def _get_item(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]
        
        # Normalize point cloud
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]
        
        return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])   # npoints
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return 'ModelNet', 'sample', (current_points, label)