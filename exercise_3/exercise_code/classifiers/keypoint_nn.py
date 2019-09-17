import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################
        self.conv1 = nn.Conv2d(1, 32, 5)
        #96--> 96-5+1=92
        self.pool1 = nn.MaxPool2d(2, 2)
        #92/2=46 ...(32,46,46)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        #46--> 46-3+1=44
        self.pool2 = nn.MaxPool2d(2, 2)
        #44/2=22
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        #22-->22-3+1=20
        self.pool3 = nn.MaxPool2d(2, 2)
        #20/2=10
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        #10-->10-3+1=8
        self.pool4 = nn.MaxPool2d(2, 2)
        #8/2=4
        
        #4x4x256
        self.fc1 = nn.Linear(4*4*256 , 1024)
        self.fc2 = nn.Linear(1024,30)
        
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
        x = self.drop4(self.pool4(F.relu(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = self.drop5(F.relu(self.fc1(x)))
        x = self.fc2(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
