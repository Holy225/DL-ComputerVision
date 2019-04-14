# -*- coding: utf-8 -*-

import torch.utils.data as data
from scipy.misc import imread
import cv2
from utils.io import unzip_objs, get_groudtruth, read_colis_results

import os
import numpy as np
import os.path


class ColisSeq(data.Dataset):
    def __init__(self, root, det_root, seq_name):
        self.root = root
        self.seq_name = seq_name
        
        self.im_root = os.path.join(self.root, self.seq_name)
        self.im_names = sorted([name for name in os.listdir(self.im_root) if os.path.splitext(name)[-1] == '.jpg'])
        
        if det_root is None:
            self.det_file = os.path.join(self.root, self.seq_name, 'identif.txt')
        else:
            self.det_file = os.path.join(det_root, '{}.txt'.format(self.seq_name))
        self.dets = read_colis_results(self.det_file)
        self.gts = get_groudtruth(self.det_file)


    def __getitem__(self, i):
        im_name = os.path.join(self.im_root, self.im_names[i])
        im = imread(im_name)  # rgb
        im = im[:, :, ::-1]  # bgr
        tlwhs = self.dets[i]
        scores = np.ones(len(tlwhs))
        
        if self.gts is not None:
            gts = self.gts.get(i, [])
            gt_tlwhs, gt_ids, _ = unzip_objs(gts)
        else:
            gt_tlwhs, gt_ids = None, None
        
        return im, tlwhs, scores, gt_tlwhs, gt_ids


    def __len__(self):
        return len(self.im_names)
    
    
def collate_fn(data):
    return data[0]


def get_loader(root, det_root, name, num_workers=0):
    
    dataset = ColisSeq(root, det_root, name) 
    data_loader = data.DataLoader(dataset, 1, False, num_workers=0, collate_fn=collate_fn)
  
    return data_loader
