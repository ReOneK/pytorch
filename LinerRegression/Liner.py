#%%
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

#读取数据
#%%
with open('F:/vscode/pytorch/pyrotch/LinerRegression/data.txt','r') as f :
    data=f.read().split('\n')
    data=[row.split(',') for row in data][:-1]
    label0=np.array([(float(row[0]),float(row[1])) for row in data if row[2]=='0'])
    label1=np.array([(float(row[0]),float(row[1])) for row in data if row[2]=='1'])
x0, y0 = label0[:, 0], label0[:, 1]
x1, y1 = label1[:, 0], label1[:, 1]
plt.plot(x0, y0, 'ro', label = 'label_0')
plt.plot(x1, y1, 'bo', label = 'label_1')
plt.legend(loc = 'best')

x=np.concatenate((label0,label1),axis=0)
x_data=torch.from_numpy(x).float()
x = np.concatenate((label0, label1), axis = 0)
x_data = torch.from_numpy(x).float()

y = [[0] for i in range(label0.shape[0])]
y += [[1] for i in range(label1.shape[0])]
y_data = torch.FloatTensor(y)



#%%
class logisticRegression(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.line = nn.Linear(2, 1)
        self.smd = nn.Sigmoid()
    def forward(self, x) :
        x = self.line(x)
        return self.smd(x)

logistic = logisticRegression()

if torch.cuda.is_available() :
    logistic.cuda()


criterion = nn.BCELoss()
# 定义优化函数为随机梯度下降(Sochastic Gradient Descent)
optimizer = torch.optim.SGD(logistic.parameters(), lr = 1e-3, momentum = 0.9)

epoches = 50000
for epoch in range(epoches) :
    if torch.cuda.is_available() :
        x = Variable(x_data).cuda()
        y = Variable(y_data).cuda()
    else :
        x = Variable(x_data)
        y = Variable(y_data)
    
    # forward 前向计算
    out = logistic(x)
    loss = criterion(out, y)
    
    # 计算准确率
    print_loss = loss.item()
    mask = out.ge(0.5).float()
    # print('size : {} - {}'.format(mask.size(), y.size()))
    correct = (mask == y).sum()
    acc = correct.item() / x.size(0)
    
    # BP回朔
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10000 == 0 :
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))
        print('loss is {:.4f}'.format(print_loss))
        print('correct rate is {:.4f}'.format(acc))
