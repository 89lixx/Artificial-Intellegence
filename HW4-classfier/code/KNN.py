from read import *
from time import *
import operator

class KNN:
    def __init__(self):
        pass
    def train(self, x , y):
        self.Xtr = x
        self.Ytr = y
    def predict(self, x, k):
        num_test = x.shape[0]
        self.x = x
        y_predict = np.zeros(num_test)
        for i in range(num_test):
            #计算距离
            distances = np.sum(np.abs(self.Xtr - self.x[i,:]) ** 2, axis=1) ** 0.5
            
            #排序
            indexes = distances.argsort()
            countDic = {}
            #前k个
            for j in range(k):
                countY = self.Ytr[indexes[j]]

                #不存在则为0
                countDic[countY] = countDic.get(countY, 0) + 1
            sortedDic = sorted(countDic.items(), key=operator.itemgetter(1), reverse=True)
            y_predict[i] = sortedDic[0][0]
            # y_predict[i] = self.Ytr[min_index]
        return y_predict


begin_time = time()

n = KNN()
# n.train(train_data[b'data'], np.array(train_data[b'labels']))
n.train(data_train, label_train)
#为了降低时间截取数组的后一部分来测试
y_predict = n.predict(data_test[:50,:],3)

accuracy = np.sum(np.array(y_predict) == label_test[:50]) / len(y_predict)

print('KNN accuracy: ',format(accuracy))

end_time = time()
print('运行时间{}秒'.format(round(end_time-begin_time,2)))