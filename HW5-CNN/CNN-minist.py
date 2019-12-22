import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data

#训练数据集
train_data = torchvision.datasets.MNIST (
    root = './',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = True,
)

#加载训练数据
load_train = Data.DataLoader (
    dataset = train_data,
    batch_size = 50,         
    shuffle = True           # 是否打乱顺序
)

#测试数据集
test_data = torchvision.datasets.MNIST (
    root = './',
    train = False,
)


# 卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.first_layer = nn.Sequential()
        #16个输出通道
        #1个输入信号通道的5*5的卷积核
        self.first_layer.add_module('Conv1', nn.Conv2d(
            in_channels = 1,   # 输入信号的通道
            out_channels = 16, # 卷积产生的通道
            kernel_size = 5,   # 卷积filter, 移动块长
            stride = 1,        # 步长
            padding = 2,       # 每一维补零的数量
        ))

        #激励层 激活函数
        self.first_layer.add_module('Relu', nn.ReLU())

        #池化层
        self.first_layer.add_module('MaxPool1', nn.MaxPool2d(kernel_size = 2))
        
        self.second_layer = nn.Sequential()
        
        #2个输出通道
        self.second_layer.add_module('Conv2',nn.Conv2d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = 5,
            stride = 1,
            padding = 2
        ))
        self.second_layer.add_module('Relu', nn.ReLU())
        self.second_layer.add_module('MaxPool2', nn.MaxPool2d(kernel_size = 2))
        
        self.layer3 = nn.Linear(1568, 10)#输出层

    # 前向传播
    def forward(self, x):
        x = self.first_layer(x)
        x = self.second_layer(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x
if __name__ == '__main__':

    test_scale = torch.unsqueeze(test_data.data, dim = 1).type(torch.FloatTensor)
    test_labels = test_data.targets

    print("开始训练")
    cnn = CNN()

    #优化器
    optimizer = torch.optim.Adam(cnn.parameters())
    
    #损失函数
    loss = nn.CrossEntropyLoss()
    for num, (X, Y) in enumerate(load_train):
        res = cnn(X)
        temp_loss = loss(res, Y)
        optimizer.zero_grad()
        temp_loss.backward()
        optimizer.step()
        if (num+1) % 200 == 0:
            print("第", num+1, "次迭代的损失为：", temp_loss.item())
    
    torch.save(cnn, "minist.pt")
    cnn = torch.load("minist.pt")
    cnn.eval()
    print("开始测试")
    result = cnn(test_scale)
    accuracy = np.mean(torch.argmax(result, 1).data.numpy() == test_labels.data.numpy())
    print("测试的精度为：", accuracy)
