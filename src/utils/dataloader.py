"""" Modified version of https://github.com/jeffwen/road_building_extraction/blob/master/src/utils/data_utils.py """
from __future__ import print_function, division
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
from natsort import natsorted
from glob import glob
import numpy as np
import os
import torch
import cv2
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_files, data_label, transform=None):
        self.data_files = np.array(data_files)
        self.data_label = np.array(data_label)
        self.transform = transform

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        #inputs = cv2.imread(self.data_files[idx])
        #label = cv2.imread(self.data_label[idx])
        #inputs = np.asarray(Image.open(self.data_files[idx]).convert('RGB'))
        #label = np.asarray(Image.open(self.data_label[idx]).convert('RGB'))
        inputs = io.imread(self.data_files[idx])
        label = io.imread(self.data_label[idx], True)
        name = str(self.data_files[idx])

        #inputs = inputs / 255.0
        #inputs = inputs.astype(np.float32)
        #label = label.astype(np.double) ## np.long --> np.double
        #print(label.ndim)
        #print(inputs.ndim)

        if label.ndim == 2:
            label = label[:,:,np.newaxis]
        if inputs.ndim == 2:
            inputs = inputs[:,:,np.newaxis]

        data = {"input":inputs, "label":label, "name":name}

        if self.transform:
            data = self.transform(data)

        return data

COLOR_LABEL = {0:(0,0,0), 1:(128,0,0)}

class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, data):
        _input, _label, _name = data["input"], data["label"], data["name"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        return {
            "input": transforms.functional.to_tensor(_input),
            #"label": torch.from_numpy(_label).unsqueeze(0).float().div(255),
            "label": torch.from_numpy(self.rgb2mask(_label)).float(),
            "name": _name
            #"map_img": torch.from_numpy((transforms.functional.to_grayscale(map_img))),
        }  # unsqueeze for the channel dimension
        
    def rgb2mask(self, rgb):
        self.rgb = rgb
        mask = np.zeros((rgb.shape[0], rgb.shape[1]))
        for k, v in COLOR_LABEL.items():
            mask[np.all(rgb==v, axis=2)] = k
        
        return mask
    



class ImageDataset(Dataset):
    """Massachusetts Road and Building dataset"""

    def __init__(self, hp, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths
            train_valid_test (string): 'train', 'valid', or 'test'
            root_dir (string): 'mass_roads', 'mass_roads_crop', or 'mass_buildings'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.train = train
        self.path = hp.train if train else hp.valid
        self.mask_list = natsorted(glob(os.path.join(self.path, "labels", "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        maskpath = self.mask_list[idx]
        
        image = io.imread((maskpath.replace(".png", ".jpg")).replace("labels", "images"))
        mask = io.imread(maskpath)
        #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        #mask0 = np.asarray(Image.open(maskpath))
        #mask1 = cv2.imread(maskpath)
        #image = np.asarray(Image.open((maskpath.replace(".png", ".jpg")).replace("labels", "images")).convert("RGB"))
        #mask = np.asarray(Image.open(maskpath).convert("RGB"))

        sample = {"input": image, "label": mask, "name": os.path.splitext(os.path.basename(maskpath))[0]}

        if self.transform:
            sample = self.transform(sample)

        return sample

class NormalizeTarget(transforms.Normalize):
    """Normalize a tensor and also return the target"""

    def __call__(self, sample):
        return {
            "sat_img": transforms.functional.normalize(
                sample["sat_img"], self.mean, self.std
            ),
            "map_img": sample["map_img"],
        }


# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
