<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/mxnet2.png width=135/> *for Deep Learning*
=====

MXNet is a deep learning framework designed for both *efficiency* and *flexibility*.
It allows you to ***mix*** the [flavours](http://mxnet.io/architecture/index.html#deep-learning-system-design-concepts) of symbolic
programming and imperative programming to ***maximize*** efficiency and productivity.
In its core, a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly.
A graph optimization layer on top of that makes symbolic execution fast and memory efficient.
The library is portable and lightweight, and it scales to multiple GPUs and multiple machines.

MXNet is also more than a deep learning project. It is also a collection of
[blue prints and guidelines](http://mxnet.io/architecture/index.html#deep-learning-system-design-concepts) for building
deep learning system, and interesting insights of DL systems for hackers.

[![Join the chat at https://gitter.im/dmlc/mxnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dmlc/mxnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

说明
-------
这是我自己移植编译的Windows版本，只是为了自我学习和与大家交流。如果有什么问题欢迎告知。<br>
CSDN:http://blog.csdn.net/sunshine_in_moon
Email:451413025@qq.com

我做的工作
-------
1、Compile im2rec.cpp to Tools.exe in Mxnet-windows\windows\x64\Release<br>
Usage:<image.lst> <image_root_dir> <output.rec> [additional parameters]<br>
Tools.exe E:\lfw\image_train.lst E:\lfw\ image.rec<br>

<<<<<<< HEAD
2、Modify tools\im2rec.py to im2rec_Linux.py for Linux and im2rec_Windows.py for Windows<br>
python im2rec_*****.py prefix root<br>
e.g. python im2rec_Windows.py E:\lfw\image E:\lfw<br>
=======
By Myself
-------
*Compile im2rec.cpp to Tools.exe in Mxnet-windows\windows\x64\Release
Usage:<image.lst> <image_root_dir> <output.rec> [additional parameters]
Tools.exe E:\lfw\image_train.lst E:\lfw\ image.rec
*Modify tools\im2rec.py to im2rec_Linux.py for Linux and im2rec_Windows.py for Windows
python im2rec_*****.py prefix root
e.g. python im2rec_Windows.py E:\lfw\image E:\lfw
>>>>>>> origin/master
