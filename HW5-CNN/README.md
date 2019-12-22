### 人工智能第五次作业

#### 一、实验内容

1. 设计并实现CNN模型，对MINIST手写字符数据集进行分类。

2. 设计并实现CNN模型，对CIFAR-10图像数据集进行分类。

#### 二、实验要求

 1.利用MNIST的训练集训练，利用MINIST的测试集测试。并完成实验代码，随实验报告一起提交。

 2.利用CIFAR-10的训练集训练，利用CIFAR-10的测试集测试。并完成实验代码，随实验报告一起提交。

#### 三、实验设计

对于CNN模型来说，其层次结构如下：

- 输入层：数据的输入
- 卷积层：使用卷积核对输入层（输入的数据）进行特征操作，一般是线性乘积求和
- 激励层：使用非线性激活函数，有：sigmoid、tanh、relu等
- 池化层：取区域平均或者最大值。
- 全连接层：实现最后的数据拟合

这五个层次中卷积层时CNN的核心。整体的设计思路其实和上次的两层神经网络差不多，只是需要对计算进行加速，主要还是使用其内部自带的函数。为了能得到更好的效果，采取双层CNN模型来处理数据集。

**1、对MINIST手写字符数据集进行分类**

- CNN模型

  对于单层CNN来说，包含上面五个层

  ```python
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
  ```

  第二层CNN，训练过程和第一层是一样的，但是多了损失函数的计算和优化器，同时损失求解后还需要向前传播更新各层参数。

  ```python
       # 前向传播
      def forward(self, x):
          x = self.first_layer(x)
          x = self.second_layer(x)
          x = x.view(x.size(0), -1)
          x = self.layer3(x)
          return x
  ```

运行结果如下：

<img src="/Users/apple/Desktop/assets/屏幕快照 2019-12-22 下午5.46.41.png" style="zoom:40%;" />

**2、对CIFAR-10图像数据集进行分类**

CIFAR-10的图像集的内容要比MINIST数据集处理起来要复杂，因为其中图片是彩色的，具有三通道，所以如果还是使用CNN对其进行分类，那么就需要更多的参数，所以这里设置了四层CNN，前三层都具有相应的结构。对于每层来说除了应有的结构外，还添加了防止过拟合的层次。那第一层为例：

```python
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
```

接下来还是上面的一样，处理向前传播的部分：

```python
    # 前向传播
    def forward(self, x):
        x = self.first_layer(x)
        x = self.second_layer(x)
        x = self.third_layer(x)
        x = x.view(x.size(0), -1)
        x = self.forth_layer(x)
        return x
```

运行结果如下：


<img src="/Users/apple/Desktop/assets/屏幕快照 2019-12-22 下午8.47.39.png" style="zoom:50%;" />

相比上一次作业而言CNN的准确度明显提高了跟多，而且在所有数据集都训练、测试的情况下，运行的速度还是很可观的。