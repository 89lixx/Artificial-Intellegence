import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

img_type = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
###################
#训练数据的设置
#数据类型
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_data = torchvision.datasets.CIFAR10(
    root = './',
    train = True,
    download = True,
    transform = transform
)

test_data = torchvision.datasets.CIFAR10(
    root = './',
    train = False,
    download = True,
    transform = transform
)

load_train = torch.utils.data.DataLoader(
    train_data,
    batch_size = 100,    
    shuffle = True
)

load_test = torch.utils.data.DataLoader(
    test_data,
    batch_size = 100, 
    shuffle = False
)

###################
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #第一层
        self.first_layer = nn.Sequential()
        self.first_layer.add_module('Conv1', nn.Conv2d(
            in_channels = 3,   # 输入信号的通道
            out_channels = 64, # 卷积产生的通道
            kernel_size = 3,   # 卷积filter
            stride = 1,        # 步长
            padding = 1,       # 每一维补零的数量
        ))
        #激励层
        self.first_layer.add_module('Relu1', nn.ReLU())
        self.first_layer.add_module('BatchNorm1', nn.BatchNorm2d(64))

        #池化层
        self.first_layer.add_module('MaxPool1', nn.MaxPool2d(kernel_size = 2))

        #防止过拟合
        self.first_layer.add_module('Dropout10',nn.Dropout(0.1))
        
        #第二层
        self.second_layer = nn.Sequential()
        self.second_layer.add_module('Conv2', nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            stride = 1,
            padding = 1,
        ))
        #激励层
        self.second_layer.add_module('Relu2', nn.ReLU())

        self.second_layer.add_module('BatchNorm2', nn.BatchNorm2d(128))
        
        self.second_layer.add_module('AvgPool1', nn.AvgPool2d(kernel_size = 2))
        
        #防止过拟合
        self.second_layer.add_module('Dropout10',nn.Dropout(0.1))
        
        self.third_layer = nn.Sequential()
        self.third_layer.add_module('Conv2', nn.Conv2d(
            in_channels = 128,
            out_channels = 256,
            kernel_size = 3,
            stride = 1,
            padding = 1,
        ))
        #激励层
        self.third_layer.add_module('Relu3', nn.ReLU())

        self.third_layer.add_module('BatchNorm3', nn.BatchNorm2d(256))
        
        self.third_layer.add_module('AvgPool2', nn.AvgPool2d(kernel_size = 8))
        
        self.third_layer.add_module('Dropout50',nn.Dropout(0.5))
        
        #第四层
        self.forth_layer = nn.Linear(256, 10);

    # 前向传播
    def forward(self, x):
        x = self.first_layer(x)
        x = self.second_layer(x)
        x = self.third_layer(x)
        x = x.view(x.size(0), -1)
        x = self.forth_layer(x)
        return x

if __name__ == '__main__':

    print("开始训练")
    cnn = CNN()
    optimizer = torch.optim.Adam(cnn.parameters())

    #损失函数
    loss = nn.CrossEntropyLoss()
    
    for num, (img, labels) in enumerate(load_train, 0):
        optimizer.zero_grad()
        res = cnn(img)
        temp_loss = loss(res, labels)
        temp_loss.backward()
        optimizer.step()
        if (num + 1) % 100 == 0:
            print("第", num + 1, "迭代的损失为：", temp_loss.item())
    
    torch.save(cnn, "cifar10.pt")
    cnn = torch.load("cifar10.pt")
    cnn.eval()
    
    print("开始测试")
    correct = 0
    correct_list = list(0. for i in range(10))
    total = 0
    total_list = list(0. for i in range(10))

    for (img, labels) in load_test:
        res = cnn(img)
        _, predicted = torch.max(res, 1)
        result = (predicted == labels).squeeze()
        for i in range(100):
            correct_list[labels[i]] += result[i].item()
            total_list[labels[i]] += 1
            correct += result[i].item()
            total += 1
    
    print("精确度为：", correct / total)
    
    for i in range(10):
        print("图片类型：%5s 的准确度为 : %.2f" %(img_type[i], correct_list[i] / total_list[i]))
