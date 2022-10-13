```
import torchvision
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

EPOCH=1
BATCH_SIZE=64
TIME_STEP=28
INPUT_SIZE=28
LR=0.1
train_data=torchvision.datasets.MNIST(root='../data/mnist',train=True,transform=torchvision.transforms.ToTensor(),
                                      download=True)

# plt.imshow(train_data.data[0].numpy(),cmap='gray')
# plt.title('%i' % train_data.targets[0])
# plt.show()

test_data=torchvision.datasets.MNIST(root='../data/mnist',train=False,transform=torchvision.transforms.ToTensor(),
                                      download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# convert test data into Variable, pick 2000 samples to speed up testing
test_x = test_data.data.type(torch.FloatTensor)[:2000]/255 # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.targets.numpy()[:2000]    # covert to numpy array

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn=nn.LSTM(input_size=INPUT_SIZE,hidden_size=64,num_layers=1,batch_first=True)
        # True:维度为这样的顺序(batch,time_step,input_size)
        self.out=nn.Linear(64,10)

    def forward(self,x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out,(h_n,h_c)=self.rnn(x,None)
        #(h_n,h_c)每一次读取input_size都会产生深层理解就是()，h_n分线，h_c主线
        #None表示第一个深层理解没有
        out=self.out(r_out[:,-1,:])
        return out

rnn=RNN()
# print(rnn)
optimizer=torch.optim.Adam(rnn.parameters())
loss_func=nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        b_x = b_x.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
```