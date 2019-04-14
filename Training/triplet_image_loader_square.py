# -*- coding: utf-8 -*-

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
from scipy import misc
import numpy as np
import os.path

import torch.utils.data
import torchvision.transforms as transforms

def default_image_loader(path):
    """
    return Image.open(path).convert('RGB')
    """
    return Pipeline(path)

class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, triplets_file_name, gpu, transform=None,loader=default_image_loader):
        """ triplets_file_name: A text file with each line containing three files, 
                For a line of files 'a b c', a triplet is defined such that image a is more 
                similar to image c than it is to image b, e.g., 
                0 2017 42 """
        triplets = []
        for line in open(triplets_file_name):
            split = line.split(" # ")
            triplets.append((split[0], split[1], split[2][:-1])) # anchor, far, close
        self.triplets = triplets
        self.transform = transform
        self.loader = loader
        self.dataset =  len(self.triplets)
        self.gpu = gpu

    def __getitem__(self, index):
        path1, path2, path3 = self.triplets[index]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        img3 = self.loader(path3)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)
    
def Pipeline(path):
    image = misc.imread(path, mode = 'RGB')
    image_pp = im_preprocess(cv2.resize(image, (128,128)))
    #return np.asarray([image_pp])
    return image_pp
    
def im_preprocess(image):
    image = np.asarray(image, np.float32)
    image -= np.array([104, 117, 123], dtype=np.float32).reshape(1, 1, -1)
    image = image.transpose((2, 0, 1))
    return image
