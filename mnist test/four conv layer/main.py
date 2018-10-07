from __future__ import print_function
import argparse
import torch
import math
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from train import train

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=1, metavar='N',
                    help='how many training processes to use (default: 2)')

def compute_entropy1(x):
    x=x[0][0]
    x=x.reshape(-1)
    G=numpy.zeros((784,784))
    A=numpy.zeros((784,784))
    for i in range(784):
        for j in range(784):
            G[i][j]=numpy.exp(-pow((x[i]-x[j]),2))
    for i in range(784):
        for j in range(784):
            A[i][j]=G[i][j]/math.sqrt(G[i][i]*G[j][j])/784
    hx=(1-0.5)*math.log(numpy.trace(A))/math.log(2.0)
    return hx

def compute_entropy2(x):
    a=numpy.ones((144,144))
    b=1
    
    for m in range(10):
        feature=x[0][m]
        feature=feature.reshape(-1)
        G=numpy.zeros((144,144))
        A=numpy.zeros((144,144))
        for i in range(144):
            for j in range(144):
                G[i][j]=numpy.exp(-pow((feature[i]-feature[j]),2))
        for i in range(144):
            for j in range(144):
                A[i][j]=G[i][j]/math.sqrt(G[i][i]*G[j][j])/144
        b=b*numpy.trace(A)
        a=numpy.multiply(a,A)
        
    
    ht1=(1-0.5)*math.log(numpy.abs(numpy.trace(a/b)))/math.log(2.0)
    return ht1

def compute_entropy3(x,y):
    hx=compute_entropy1(x)
    ht1=compute_entropy2(y)
    hxt1high=hx+ht1
    hxt1low=max(hx,ht1)
    return hxt1high,hxt1low

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)



    def forward1(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2(x), 3,1,1))

        x = F.relu(F.max_pool2d(self.conv3(x), 3,1,1))
        
        x = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(x)),3,1,1))
       
        x = x.view(-1, 1440)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def forward(self,x):
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    

    def forward(self,x):
        #print(x.size())
        temp1=x.detach().numpy()
        #hx=compute_entropy1(temp1)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        temp2=x.detach().numpy()
        #ht1=compute_entropy2(temp2)
        #hxt1high,hxt1low=compute_entropy3(temp1,temp2)
        
        #Ixt1=hx+ht1-max(hx,ht1)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        temp3=x.detach().numpy()
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1),temp1,temp2


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model = Net()
    model1= Net1()
    model.share_memory() # gradients are allocated lazily, so they are not shared here
    model1.share_memory()
    #model = model.cuda()
    #cudnn.benchmark = True
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model,model1))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

