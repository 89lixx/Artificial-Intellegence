from read import *
from time import *

class NeuralNetwork:
    def __init__(self,input, output, hidden):
        self.layers = {'hidden_weight':0.0001 * np.random.randn(input, hidden), 'hidden_data':np.zeros(hidden), 'output_weight':0.0001 * np.random.randn(hidden, output), 'output_data':np.zeros(output)}
    def loss(self, train_data, train_label, reg):
        #从字典中分别取出对应的layer和数据
        hidden_weight = self.layers['hidden_weight']
        hidden_data = self.layers['hidden_data']
        output_weight = self.layers['output_weight']
        output_data = self.layers['output_data']

        data_scale = train_data.shape[0]
        #前向传播
        forward=np.maximum(0,np.dot(train_data,hidden_weight)+hidden_data)
        forward_output=np.dot(forward,output_weight)+output_data
        #计算损失
        #这里类似于SVM
        #需要根据前向传播的值来进行类别的归属判断
        loss_list = np.exp(forward_output[np.arange(data_scale), train_label])
        loss = sum(-np.log(loss_list / np.sum(np.exp(forward_output), axis=1))) / data_scale + (np.sum(output_weight**2) * reg + np.sum(hidden_weight**2))
        
        #反向传播
        #输出层梯度计算
        output_grad = np.exp(forward_output) / np.sum(np.exp(forward_output), axis=1).reshape(data_scale, 1)
        output_grad[np.arange(data_scale), train_label] -= 1
        output_grad = output_grad / data_scale
        #权重矩阵
        output_data_grad = np.sum(output_grad, axis=0)
        output_matrix_grad = output_weight * 2 * reg  + np.dot(forward.T, output_grad)
        
        #隐藏层梯度
        forward_grad = np.dot(output_grad, output_weight.T)
        forward_grad[forward<=0] = 0
        #权重矩阵
        hidden_data_grad = np.sum(forward_grad, axis=0)
        hidden_matrix_grad = hidden_weight * 2 * reg + np.dot(train_data.T, forward_grad)
        
        #计算最终梯度
        #由于包含四部分所有使用字典来存放
        grads = {}
        grads['output_weight'] = output_matrix_grad
        grads['output_data'] = output_data_grad
        grads['hidden_weight'] = hidden_matrix_grad
        grads['hidden_data'] = hidden_data_grad
        return loss, grads
    
    def train(self, train_data, train_label, iterations , batch_size , learning_rate=8e-4 , reg=1.0 , decay_rate=0.95):
        data_scale = train_data.shape[0]
        
        reduce_rate = max(data_scale / batch_size, 1)
        #类似于SVM，抽取一部分样本进行更新权重
        for i in range(iterations):
            
            indexes = np.random.choice(data_scale, batch_size)
            #更新
            loss, grads = self.loss(train_data[indexes], train_label[indexes], reg)
            #得到的对应层的权重更新本层数据
            for layer in self.layers:
                self.layers[layer] = self.layers[layer] - learning_rate * grads[layer]
            #学习率衰减
            if i % reduce_rate == 0:
                learning_rate *= decay_rate
    

    def predict(self, test_data):
        y_predict = np.zeros(test_data.shape[0])
        hidden_weight= self.layers['hidden_weight']
        hidden_data = self.layers['hidden_data']
        output_weight= self.layers['output_weight']
        output_data = self.layers['output_data']
        forward = np.maximum(0, np.dot(test_data, hidden_weight) + hidden_data)
        forward_output = np.dot(forward, output_weight) + output_data
        y_predict = np.argmax(forward_output, axis=1)
        return y_predict

begin_time = time()
network = NeuralNetwork(input=3072, hidden=100, output=10)
network.train(data_train[:8000,:], label_train[:8000], iterations=2000, batch_size=200)
y_predict = network.predict(data_test)
accuracy = np.sum(np.array(y_predict)==label_test) / len(y_predict)
print("二层神经网络准确率:", accuracy)
end_time = time()
print('二层神经网络运行时间{}秒'.format(round(end_time-begin_time, 2)))


