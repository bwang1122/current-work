import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy
import math
from torchvision import datasets, transforms

def gaussian_matrix(x):
    variable1=numpy.dot(x,x.T)
    G=2*variable1
    C=numpy.zeros((200,200))
    for j in range(200):
        for i in range(200):
            C[i][j]=G[i][j]-variable1[j][j]
    for i in range(200):
        for j in range(200):
            C[i][j]=C[i][j]-variable1[i][i]
    
    for i in range(200):
        for j in range(200):
            C[i][j]=numpy.exp(1/(2*(200**(-1/148)))*C[i][j])
    
    return C

def compute_entropy1(x):
    
    variable1=numpy.dot(x,x.T)
    
    G=2*variable1
    
    C=numpy.zeros((200,200))
    for j in range(200):
        for i in range(200):
            C[i][j]=G[i][j]-variable1[j][j]
    for i in range(200):
        for j in range(200):
            C[i][j]=C[i][j]-variable1[i][i]
    #print(C)
    for i in range(200):
        for j in range(200):
            C[i][j]=numpy.exp(1/(2*(200**(-1/788)))*C[i][j])
    lamda,feature=numpy.linalg.eig(C)
    
    traceA=0
    
    for i in range(200):
        traceA=traceA+pow(numpy.abs(lamda[i]),2)
    
    hx=-1*math.log(traceA)/math.log(2.0)
    return hx,C

def compute_entropy2(x):
    a=numpy.ones((200,200))   
    b=1
    for m in range(10):
        featureG=numpy.zeros((200,200))
        featureG=gaussian_matrix(x[:,m,:])
        a=numpy.multiply(a,featureG)
        
    b=numpy.trace(a)
    lamda,feature=numpy.linalg.eig(a/b)
    
    traceA=0
    #print(G)
    for i in range(200):
        traceA=traceA+pow(numpy.abs(lamda[i]),2)
    
    ht1=-1*math.log(numpy.abs(traceA))/math.log(2.0)
    return ht1,a,b

def compute_mutual(x,y,tr,hx,ht):
    d=numpy.multiply(x,y)
    e=d/numpy.trace(d)
    lamda1,feature1=numpy.linalg.eig(e)
    traceA=0
    
    for i in range(200):
        traceA=traceA+pow(numpy.abs(lamda1[i]),2)
    
    hxt=1*math.log(numpy.abs(traceA))/math.log(2)
    mutualxt=hx+ht-hxt
    return mutualxt

def train(rank, args, model,model1):
    torch.manual_seed(args.seed + rank)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor()
                    ])),
        batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, train_loader, optimizer)
        torch.save(model.state_dict(), 'modeltest.pth')
        model1.load_state_dict(torch.load('modeltest.pth'))
        test_epoch(model1, test_loader)
        
    


def train_epoch(epoch, args, model, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    count=1
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        #data=data.cuda()
        #target=target.cuda()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        count=count+1
        
        if count==100:
            break
        if batch_idx % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))


def test_epoch(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    count=0
    inputx=numpy.zeros((200,784))
    inputt1=numpy.zeros((200,10,144))
    inputt2=numpy.zeros((200,10,144))
    inputt3=numpy.zeros((200,10,144))
    inputt4=numpy.zeros((200,10,144))   
    with torch.no_grad():
        for data, target in data_loader:
            output,temp,temp1,temp2,temp3,temp4= model(data)
            
            temp=temp.reshape(-1)
            
            inputx[count]=temp
            for m in range(10):
                inputt1[count][m]=temp1[0][m].reshape(-1)
                
                
                inputt2[count][m]=temp2[0][m].reshape(-1)   
                inputt3[count][m]=temp3[0][m].reshape(-1)
                inputt4[count][m]=temp4[0][m].reshape(-1)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target).sum().item()
            count=count+1
            if count==200:
                break
    hx,matrixx=compute_entropy1(inputx)
    
    ht1,matrix1,trm1=compute_entropy2(inputt1)
    mutualxt1=compute_mutual(matrixx,matrix1,trm1,hx,ht1)
    ht2,matrix2,trm2=compute_entropy2(inputt2)
    mutualxt2=compute_mutual(matrixx,matrix2,trm2,hx,ht2)
    ht3,matrix3,trm3=compute_entropy2(inputt3)
    mutualxt3=compute_mutual(matrixx,matrix3,trm3,hx,ht3)
    ht4,matrix4,trm4=compute_entropy2(inputt4)
    mutualxt4=compute_mutual(matrixx,matrix4,trm4,hx,ht4)
    print(mutualxt1)
    print(mutualxt2)
    print(mutualxt3)
    print(mutualxt4)
    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    