```
if __name__ == '__main__':
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    root = '../data/'
    batch_size=256
    mnist_train=torchvision.datasets.FashionMNIST(root=root,train=True,transform=torchvision.transforms.ToTensor(),download=True)
    # mnist_test=torchvision.datasets.FashionMNIST(root=root,train=False,transform=torchvision.transforms.ToTensor(),download=True)
    train_iter=torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    # test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=4)

    class Flatten(nn.Module):
        def __init__(self):
            super(Flatten,self).__init__()

        def forward(self,x):
            return x.view(x.shape[0],-1)


    net=nn.Sequential(
        Flatten(),
        nn.Linear(num_inputs,num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens,num_outputs)
    )

    for params in net.parameters():
        init.normal_(params,mean=0,std=0.01)

    loss=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(net.parameters(),lr=0.5)
    num_epoch=5

    for epoch in range(num_epoch):
        train_l_sum=0
        train_acc_sum=0
        n=0
        for X,y in train_iter:
            y_hat=net(X)
            l=loss(y_hat,y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum+=l.item()
            train_acc_sum +=(y_hat.argmax(dim=1)==y).sum().item()

            n=n+y.shape[0]
            print('n %d,epoch %d loss %.4f, train acc %.3f '
                  % ( n+0 ,epoch+1, train_l_sum / n, train_acc_sum / n))
```