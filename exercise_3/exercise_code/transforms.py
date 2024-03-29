import torch
from torchvision import transforms, utils
# tranforms
import numpy as np 

class Normalize(object):
    """Normalizes keypoints.
    """
    def __call__(self, sample):
        
        image, key_pts = sample['image'], sample['keypoints']
        
        ##############################################################
        # TODO: Implemnet the Normalize function, where we normalize #
        # the image from [0, 255] to [0,1] and keypoints from [0, 96]#
        # to [-1, 1]                                                 #
        ##############################################################        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        image_copy=  image_copy/255.0
        key_pts_copy = (key_pts_copy- 48)/48.0
        ##############################################################
        # End of your code                                           #
        ##############################################################
        return {'image': image_copy, 'keypoints': key_pts_copy}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        image, key_pts = sample['image'], sample['keypoints']
        return {'image': torch.from_numpy(image).float(),
                'keypoints': torch.from_numpy(key_pts).float()}