from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg
import pandas as pd
import os
import numpy as np
from .data_utils import get_image,get_keypoints


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            custom_point (list): which points to train on
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.key_pts_frame.dropna(inplace=True)
        self.key_pts_frame.reset_index(drop=True, inplace=True)
        self.transform = transform
        
    def __len__(self):
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataset                                     #
        ########################################################################
        return len(self.key_pts_frame)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def __getitem__(self, idx):
        ########################################################################
        # TODO:                                                                #
        # Return the idx sample in the Dataset. A simple should be a dictionary#
        # where the key, value should be like                                  #
        #        {'image': image of shape [C, H, W],                           #
        #         'keypoints': keypoints of shape [num_keypoints, 2]}          #
        ########################################################################

        img_list = self.key_pts_frame.iloc[idx]['Image']
        image= np.array([int(item) for item in img_list.split()]).reshape((1,96, 96))
        
        keypoint_cols = list(self.key_pts_frame.columns)[:-1]
        keypoints = self.key_pts_frame.iloc[idx][keypoint_cols].values.reshape((15, 2)).astype(np.float32)
        
        
        sample = { 'image':image, 'keypoints':keypoints}
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        if self.transform:
            sample = self.transform(sample)

        return sample
    
