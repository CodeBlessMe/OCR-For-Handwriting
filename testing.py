# **********
# Python project: OCR for handwriting
# Members: Robin, Madina, Jiang
# December 2021
#
# This python file is used to validation the accuracy of the model.
# **********

# --------------------load data------------------------------------
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN
from PIL import Image, ImageEnhance 
import pathlib
import matplotlib.pyplot as plt


test_data = dataset.EMNIST(root = torch.__path__[0]  + "\PyCharm\OCR\PyCharm\OCR",
                           train=False, transform=transforms.ToTensor(),
                           split="balanced", download=False) # True if not yet on your computer

test_loader = data_utils.DataLoader(dataset = test_data,
                                    batch_size=10, shuffle=True)

# --------------------load model-----------------------------------------
cnn = torch.load(str(pathlib.Path().resolve()) + "\deepLearnModel")
if torch.cuda.is_available():
        cnn = cnn.cuda()

# --------------------on test set-----------------------------------------
loss_test = 0
accuracy_test = 0

for i, (images, labels) in enumerate(test_loader):
        # Using GPU (if available)
        if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda() 
        outputs = cnn(images)
        _, pred = outputs.max(1) # get the index with the highest probability
        accuracy_test += (pred == labels).sum().item()
accuracy_test = str(round(100*accuracy_test / len(test_data)))
print("Test set accuracy: " + accuracy_test + "%")
        
# --------------------on our own images-----------------------------------------

# pre-processing of png to make it compatible for the CNN   
def pngToInputTensor(filename):
        # 1: convert to 0-1 grayscale 1x28x28 pytorch tensors
        tensor_trans = transforms.ToTensor()    
        image = Image.open(filename).convert('L').resize( (28,28), Image.ANTIALIAS)
        
        # needed to adjust the values to make in 0-1 range with the background = 0
        enhancer = ImageEnhance.Contrast(image) # to make more similar to EMNIST: increase contrast
        image_enhanced = enhancer.enhance(6.0)            
        image_tensor = (-tensor_trans(image_enhanced) + 1).type(torch.float) 
            
        
        #notice in EMIST all background has been fully blacked out
        for rowindex, row in enumerate(image_tensor[0,:,:]):
                for colindex, element in enumerate(row):
                        if element < 0.6:
                                image_tensor[0,rowindex, colindex] = 0

        image_tensor = torch.rot90(image_tensor, 1, [1, 2])
        image_tensor = torch.fliplr(image_tensor)

        return image_tensor

class Dataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
                self.features = features  
                self.labels = labels
                
        def __len__(self):
                return len(self.labels)
    
        def __getitem__(self, idx):
                return (self.features[idx], self.labels[idx])


# prepare the own images set by reading images.txt and fetching/pre-processing images + their labels
image_labels = []
image_tensors = []
with open("images.txt") as file:
        for line in file:
                if line[0] == "#":
                        continue # header
                linelist = line.rstrip().split(",")
                name = "img" + linelist[0] + ".jpg"
                image_tensors.append( pngToInputTensor(name) ) 
                image_labels.append(linelist[1])
                
image_dataset = Dataset(image_tensors, image_labels)
        
image_loader = data_utils.DataLoader(dataset = image_dataset,
                                            batch_size = 1, shuffle=False)
cnn.cpu()
accuracy_own = 0
print("----------------------------------------------")
print("The mistakes made by the predictions are:...")
for i, (image, label) in enumerate(image_loader):
        label = label[0] # batches is a tuple but only need first element

        output = cnn(image)
        _, pred = output.max(1)
        pred = test_data.classes[pred] # fetch corresponding label string to compare with the label
        # uncomment this section if you want to see the own images + predicted/true label
        
        figure = plt.figure(i)
        image = image.cpu()
        plt.imshow(image[0,0,:,:], cmap = "gray") #[0,0,:,:]
        plt.title("Real label: " + label + ", predicted label: " + pred)
        

        if pred == label:
                accuracy_own += 1
        else: 
                #  to see what errors are made
                print("- Index: " + str(i+1) + ", Real label: " + label + ", Predicted label: " + pred) 
                continue 
                


plt.show()
accuracy_own = str(round(100*accuracy_own / len(image_labels)))
print("----------------------------------------------")
print("Own images accuracy: " + accuracy_own + "%")