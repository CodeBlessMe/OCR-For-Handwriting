# **********
# Python project: OCR for handwriting
# Members: Robin, Madina, Jiang
# December 2021
#
# This python file is the Convolutional neural network model.
# **********

import torch

class CNN(torch.nn.Module):
    # Inherit from super class
    def __init__(self):
        super(CNN, self).__init__()
        # The input image is 1*28*28
        self.layer1 = torch.nn.Sequential(
            # 1 input channel, 16 output channels
            torch.nn.Conv2d(1, 16, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2) 
        )
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        
        # Define a fully connected layer
        # 47: there are 47 classes        
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 64, 47),
        )

    def forward(self, x): 
        out = self.layer1(x) 
        out = self.layer2(out)
        out = self.layer3(out) 
        # Modify the shape
        out = out.view(out.size()[0], -1) 
        out = self.layer4(out)
        return out