```
dataset=torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),download=False)
dataloader=DataLoader(dataset,batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model1=nn.Sequential(
            nn.Conv2d(3, 32, 5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
    def forward(self,x):
        x=self.model1(x)
        return x

net=Net()
# print(net)
input=torch.ones((64,3,32,32))
out=net(input)
print(out.shape)
```