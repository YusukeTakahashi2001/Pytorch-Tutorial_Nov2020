#import Modules
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
#-----------------------------------
#Datasets: FashionMNIST
#Read
fashion_mnist_data = torchvision.datasets.FashionMNIST(
    './data',
    transform=torchvision.transforms.ToTensor(),
    download=True)
data_loader = torch.utils.data.DataLoader(
dataset= fashion_mnist_data,
batch_size = 16,
shuffle=True)
fashion_mnist_data_test = torchvision.datasets.FashionMNIST(
'.data',
transform=torchvision.transforms.ToTensor(),
train=False,
download=True)
data_loader_test = torch.utils.data.DataLoader(
dataset=fashion_mnist_data_test,
batch_size=16,
shuffle=True)
#-----------------------------------------------

#model
class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1,20,5,1)
        self.conv2 = nn.Conv2d(20,50,5,1)
        self.aff1 = nn.Linear(4*4*50,500)
        self.aff2 = nn.Linear(500,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        #Tensor の変形
        x = x.view(-1,4*4*50)#to affine1
        x = F.relu(self.aff1(x))#to affine2
        x = self.aff2(x)
        #softmax (dimension=1)
        out = F.log_softmax(x,dim=1)
        return out

#hyperparam
using_cuda = torch.cuda.is_available()#boolean value
net = CNN()
optimizer = optim.SGD(net.parameters(),lr=0.01)
criterion = nn.NLLLoss()

if using_cuda:
    net.cuda()
    criterion.cuda()

accuracies = []
epochs =40

for i in range(epochs):
    #train--------------------------
    #devide all data to batch and labels.
    for batch,labels in data_loader:
        #if cuda is available,use cuda about Auto Gradients.
        if using_cuda:
            x = Variable(batch.cuda())
            y = Variable(labels.cuda())
        else:
            x = Variable(batch)
            y = Variable(labels)
        #Initialize all gradients to zero for each epoch.
        optimizer.zero_grad()
        #train model
        output = net(x)
        #Calcurate loss.
        loss = criterion(output,y)
        #backward
        loss.backward()
        #Update params
        optimizer.step()

    n_true = 0
    #test--------------------------------
    for batch,labels in data_loader_test:
        #Read Grad from Train data
        if using_cuda:
            output = net(Variable(batch.cuda()))
        else:
            output = net(Variable(batch))
            _,predicted = torch.max(output.data,1)
        if using_cuda:
            y_predicted = predicted.cpu().numpy()
        else:
            y_predicted = predicted.numpy()
            n_true += np.sum(y_predicted == labels.numpy())

    total = len(fashion_mnist_data_test)
    accuracy = 100 * n_true/total
    accuracies.append(accuracy)

print('Max:',max(accuracies),'Min:',min(accuracies))
