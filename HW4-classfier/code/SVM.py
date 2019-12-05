from time import *
from read import *
import numpy as np
class SVM:
    def __init__(self):
        #初始接下来三个函数都要用到的变量
        self.matrix = None
    def loss(self, train_data, train_label, reg):
        train_data_scale = train_data.shape[0]
        
         #不同类别的分数
        scores = np.dot(train_data,self.matrix)
        #变成秩为1的分数矩阵
        correct_socres = scores[np.arange(train_data_scale), train_label].reshape(train_data_scale, 1)
        
        #计算样本的损失
        loss_list = scores - correct_socres + 1
         #将正确的分类变成0，便是没有损失
        loss_list[np.arange(train_data_scale), train_label] = 0

        #计算样本的整体损失
        loss = np.sum(loss_list[loss_list > 0]) / train_data_scale
        loss = loss + np.sum(self.matrix * self.matrix) * 0.5 * reg
        
        #再次修改分数，分数比正确分类大的分成1，否则为0
        loss_list = np.minimum(1, loss_list)
        loss_list = np.maximum(0, loss_list)
        
        wrong_num = np.sum(loss_list, axis=1)
        loss_list[np.arange(train_data_scale), train_label] = - wrong_num
        #计算梯度
        grad = np.zeros(self.matrix.shape)
        grad = grad + np.dot(train_data.T, loss_list) / train_data_scale + reg * self.matrix
        
        return loss, grad
    

    def train(self, train_data, train_label, class_num, learning_rate=1e-3, iterations=1000,batch_size=100, reg=1e-5):
        #初始化
        train_scale,sample_size = train_data.shape
        self.matrix = np.random.randn(sample_size, class_num) * 0.001

        #梯度下降
        for i in range(iterations):
            #随机抽取样本来进行更新
            indexes = np.random.choice(train_scale, batch_size)
            loss,grad = self.loss(train_data[indexes], train_label[indexes], reg)
            self.matrix = self.matrix - learning_rate * grad
    def predict(self, test_data):
        y_predict = np.zeros(test_data.shape[0])
        scores = np.dot(test_data, self.matrix)
        y_predict = np.argmax(scores, axis=1)
        return y_predict

begin_time = time()
svm = SVM()
svm.train(data_train, label_train, class_num=10)
y_predict = svm.predict(data_test)
accuracy = np.sum(np.array(y_predict) == label_test) / len(y_predict)
print('SVM的准确度为:',accuracy)
end_time = time()
print('运行时间{}秒'.format(round(end_time-begin_time, 2)))