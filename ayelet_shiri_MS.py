from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils import data
import os
import glob
import os.path as osp
from PIL import Image
import csv
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
path='./MS_Dataset_2019/training'
path_val='./MS_Dataset_2019/validation'
print (path)

class MSDataset(data.Dataset):
    def __init__(self, dataPath, labelsFile, transform=None):
        """ Intialize the dataset
        """
        self.dataPath=dataPath
        self.transform=transform

        with open(os.path.join(self.dataPath,labelsFile)) as f:
            # for line in csv.reader(f):
            #     self.imageNames = line[0]
            #     self.labels=line[1]

            self.labels=[tuple(line) for line in csv.reader(f)]


        # for i in range(len(self.labels)):
        #     assert os.path.isfile(dataPath+'/'+str(self.labels[i][0]))

    # You must override __getitem__ and __len__
    def __getitem__(self, idx):
        imageName, imageLabel=self.labels[idx][0:]
        imagePath=os.path.join(self.dataPath,imageName)
        image = Image.open(open(imagePath,'rb'))

        if self.transform:
            image=self.transform(image)
        return (image,imageLabel)

    def __len__(self):
        return len(self.labels)


trainset=MSDataset(path,'MSdata.csv',transform=transforms.ToTensor())


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)

valset=MSDataset(path_val,'MSdata_val.csv',transform=transforms.ToTensor())


valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                          shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() #imsize=32*32
        self.conv1 = nn.Conv2d(1, 6, 5) #(in channels, out channels, kernel size) (28*28)
        self.pool = nn.MaxPool2d(2, 2) #(kernel size) (14*14)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #32*32-->28*8-->14*14
        x = self.pool(F.relu(self.conv2(x)))#14*14-->10*10-->5*5
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def main():
    train_losses=[]
    val_losses=[]
    accuracy_list=[]
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0): []
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels_list=[]
            for k in labels:
                k=int(k)
                labels_list.append(k)

            labels=torch.tensor([labels_list])
            labels = labels.squeeze()

            #print (inputs, "type inputs:",type(inputs), labels,  "type inputs:",type(labels))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:    # print every 2000 mini-batches
                val_loss=0
                accuracy=0
                net.eval()
                with torch.no_grad():
                    for inputs, labels in valloader: #runs through ALL images
                        labels_list = []
                        for k in labels:
                            k = int(k)
                            labels_list.append(k)
                        labels = torch.tensor([labels_list])
                        labels = labels.squeeze()
                        outputs=net(inputs)
                        outputs=outputs.exp()
                        batch_loss=criterion(outputs, labels)
                        val_loss += batch_loss.item()
                        max_score, max_class=outputs.topk(1,dim=1)
                        equals=max_class==labels
                        accuracy+=torch.mean(equals.type(torch.FloatTensor)).item() #sums accuracy, adding each bach's accuracy every time
                accuracy_list.append(accuracy/len(valloader))
                print ("accuracy: ", accuracy_list, "epoch:", epoch)
                train_losses.append(running_loss/len(trainloader))
                val_losses.append(val_loss/len(valloader))
                print ("Epoch: ",epoch, "Train losses: ", train_losses[epoch], "Val_losses: " ,val_losses[epoch])
                net.train()
                running_loss=0

    print('Finished Training')

    torch.save(net.state_dict(), './MS_Dataset_2019/validation/param')
if __name__ == '__main__':
    main()

a=3