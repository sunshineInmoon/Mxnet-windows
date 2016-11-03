# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 19:07:06 2016

@author: zhanghua
"""

import mxnet as mx

#首先定义一个数据迭代器
data_dir = "E:/Mxnet/mxnet/example/image-classification/mnist/"
train = mx.io.MNISTIter(
        image = data_dir + "train-images-idx3-ubyte",
        label = data_dir + "train-labels-idx1-ubyte",
        batch_size = 128,
        shuffle = True,
        flat = False,
        input_shape = (1,28,28))

val = mx.io.MNISTIter(
        image = data_dir + "t10k-images-idx3-ubyte",
        label = data_dir + "t10k-labels-idx1-ubyte",
        batch_size = 128,
        input_shape = (1,28,28),
        flat = False)
        

"""
#定义一个网络
#输入层
data = mx.symbol.Variable("data")
#第一个卷基层
conv1 = mx.symbol.Convolution(data=data,num_filter=20,kernel=(5,5))
relu1 = mx.symbol.Activation(data=conv1,act_type="relu")
pool1 = mx.symbol.Pooling(data=relu1,pool_type="max",kernel=(2,2),stride=(2,2))
#第二个卷基层
conv2 = mx.symbol.Convolution(data=pool1,kernel=(5,5),num_filter=50)
relu2 = mx.symbol.Activation(data=conv2,act_type="relu")
pool2 = mx.symbol.Pooling(data=relu2,pool_type="max",kernel=(2,2),stride=(2,2))
#第一个全连接层
#第一个全连接层要做一个平滑
flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten,num_hidden=500)
relu3 = mx.symbol.Activation(data=fc1,act_type="relu")
#第二个卷积层
fc2 = mx.symbol.FullyConnected(data=relu3,num_hidden=10)
#损失函数
lenet = mx.symbol.SoftmaxOutput(data=fc2,name="softmax")
#保存网路
#lenet.save(data_dir+"mnist_net.json")
"""
#从网络配置文件****。josn文件中加载一个网络
lenet = mx.symbol.load('E:/Deep Learning/mxnet/myself_mnist/myself_mnist-symbol.json')

#训练网络
#logging
import logging
head = '%(asctime)-15s Node[0] %(message)s'
logging.basicConfig(level=logging.DEBUG,format=head)
logging.info('start with arguments %s','It is kaishi!')

prefix="myself_mnist"
model = mx.model.FeedForward(
        ctx = mx.gpu(0),
        momentum = 0.9,
        wd = 0.00001,
        initializer = mx.init.Xavier(factor_type="in",magnitude=2.34),
        symbol = lenet,
        num_epoch = 10,
        epoch_size = 468,
        learning_rate = .1)
        
## TopKAccuracy only allows top_k > 1
eval_metrics = ['accuracy']
for top_k in [5, 10, 20]:
    eval_metrics.append(mx.metric.create('top_k_accuracy', top_k = top_k))
print u'开始训练'
batch_end_callback = []
batch_end_callback.append(mx.callback.Speedometer(128,50))
model.fit(
        X=train,
        eval_data=val,
        eval_metric = eval_metrics,
        batch_end_callback = batch_end_callback,
        epoch_end_callback = None)
print u'训练结束，保存model'
model.save(prefix,10)