"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        feats = list(models.vgg16(pretrained=True).features.children())

        self.feats = nn.Sequential(*feats[0:17])
        self.pool4 = nn.Sequential(*feats[17:24])
        self.pool5 = nn.Sequential(*feats[24:])
        self.res_pool4 = nn.Conv2d(512, num_classes, 1)
        
        self.fconn = nn.Sequential(nn.Conv2d(512, 4096, 7, padding=3),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(4096, 4096,1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(4096, num_classes, 1)
                               )
    
        self.activation = nn.Sigmoid()
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        feats = self.feats(x)
        pool4 = self.pool4(feats)
        pool5 = self.pool5(pool4)
        
        fconn = self.fconn(pool5)
        res_pool4 = self.res_pool4(pool4)
        
        up_fconn = F.upsample(fconn, pool4.size()[2:], mode='bilinear')
        out = up_fconn + res_pool4
        
        upsample = F.upsample(out, x.size()[2:],mode='bilinear')
    
        out = self.activation(upsample)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return out

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
