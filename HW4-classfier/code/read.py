import pickle
import numpy as np

file_path = './data/data_batch_'
test_file = './data/test_batch'
#读取单个文件数据
def load_single_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    fo.close()
    return data



#将读取的多个文件数据存放到一起
def load_file(file):
    data_train = []
    label_train = []
    #融合训练数据集
    res_data = {}
    for i in range(1,6):
        data = load_single_file(file_path+str(i))
        res_data.update(data)
        for d in data[b'data']:
            data_train.append(d)
        for l in data[b'labels']:
            label_train.append(l)
    # print(res_data[b'data'].shape,'aaa')
    # print(type(res_data[b'data']))
    # print(type(res_data[b'labels']))
    #测试数据集
    data_test = []
    label_test = []
    data = load_single_file(test_file)
    for i in data[b'data']:
        data_test.append(i)
    for i in data[b'labels']:
        label_test.append(i)
    return (np.array(data_train), np.array(label_train), np.array(data_test), np.array(label_test))
    # return (data_train, np.array(label_train), data_test, np.array(label_test))


data_train,label_train,data_test,label_test = load_file(file_path)
# print('训练数据规模：',data_train.shape)
# print('训练标签规模：',label_train.shape)
# print('测试数据规模：',data_test.shape)
# print('测试标签规模：',label_test.shape)