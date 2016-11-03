# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 14:32:37 2016

@author: zhanghua
"""

import mxnet as mx

#首先定义一个训练数据迭代器和一个验证数据迭代器
data_dir = 'E:/Mxnet/mxnet/example/image-classification/mnist/'
train = mx.io.MNISTIter(
        image = data_dir + 'train-images-idx3-ubyte',
        label = data_dir + 'train-labels-idx1-ubyte',
        batch_size = 128,
        shuffle = True,
        flat = False,
        input_shape = (1,28,28))
        
val = mx.io.MNISTIter(
        image = data_dir + 't10k-images-idx3-ubyte',
        label = data_dir + 't10k-labels-idx1-ubyte',
        batch_size = 128,
        input_shape = (1,28,28),
        flat = False)
        
#从网络配置文件****。josn文件中加载一个网络
lenet = mx.symbol.load('E:/Deep Learning/mxnet/myself_mnist/myself_mnist-symbol.json')

#训练网络
import logging
head = '%(asctime)-15s Node[0] %(message)s'
logging.basicConfig(level=logging.DEBUG,format=head)
logging.info('start with arguments %s','It is kaishi!')

preifx = 'model/myself_mnist2'
model = mx.model.FeedForward(
        ctx = mx.gpu(0),
        learning_rate = 0.1,
        momentum = 0.9,
        wd = 0.00001,
        initializer = mx.init.Xavier(factor_type='in',magnitude=2.34),
        symbol = lenet,
        num_epoch = 10,
        epoch_size = 468)
        
eval_metrics = ['accuracy']
for top_k in [5,10,20]:
    eval_metrics.append(mx.metric.create('top_k_accuracy',top_k = top_k))
    
batch_end_callback = []
batch_end_callback.append(mx.callback.Speedometer(128,50))
checkpoint = mx.callback.do_checkpoint('model/myself_mnist2')
model.fit(
        X=train,
        eval_data=val,
        eval_metric = eval_metrics,
        batch_end_callback = batch_end_callback,
        epoch_end_callback = checkpoint)