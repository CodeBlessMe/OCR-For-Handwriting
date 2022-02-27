# **********
# Python project: OCR for handwriting
# Members: Robin, Madina, Jiang
# December 2021
#
# This python file is used to train the model and save the model locally.
# **********

import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN
import pathlib

# **********
# Load data
#
# Root is the path to store and load data
# Transform transforms the data to tensor.
# For EMNIST, there are several split options, 'balanced' has 131600 pictures, 47 classes
#       every class has 2400 pictures for training and 400 pictures for testing.
# If you don't have data locally, change the download to True.
# **********

# Load train data
train_data = dataset.EMNIST(root = torch.__path__[0]  + "\PyCharm\OCR",
                            train=True, transform=transforms.ToTensor(),
                            split="balanced", download=False) # True if not yet on your computer

# **********
# We use mini batch, the batch size is 64. And we also shuffle data to remove the influence of order,
#       and to make data of each batch more abundant.
# DataLoader can extract data cyclically, so the amount of data will not limit the training times.
# **********

train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=64, shuffle=True)

# Initializing the network
cnn = CNN()
# Using GPU to do training (if available)
if torch.cuda.is_available():
    cnn = cnn.cuda()


# Cross entropy loss is the best loss function for classification.
# In pytorch, CrossEntropyLoss() converts the output results using sigmoid first,
#       and then puts them into the traditional cross entropy function.
loss_func = torch.nn.CrossEntropyLoss()

# Optimizer using Adam, because it is not sensitive to learn rate.
# the learn rate is 0.01
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)



# **********
# Training
#
# Each epoch trains the model using all train samples
# **********
for epoch in range(10):
    accuracy = 0    
    for i, (images, labels) in enumerate(train_loader):
        # Using GPU (if available)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = cnn(images)
        loss = loss_func(outputs, labels)
        
        # The output is a series of probabilities
        _, pred = outputs.max(1)
        # Sum the samples with correct predication
        accuracy += (pred == labels).sum().item()  
        
        # Optimizing parameters by backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
        
    accuracy = accuracy / len(train_data)
    print("epoch is " + str(epoch +  1) + ", learning accuracy is " + str(round(100*accuracy)) + "%") 

# Save model
torch.save(cnn, str(pathlib.Path().resolve()) + "\deepLearnModel") # saving in current working dir
print("Finished training and saving model")