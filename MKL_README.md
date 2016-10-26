# MKL2017 PLUGIN

MKL2017 is one INTEL released library to accelerate Deep Neural Network (DNN) applications on Intel architecture.
This README shows user how to setup and install MKL2017 library with mxnet.


## Build/Install Instructions:
```
Download MKL:

```

## Build/Install MxNet
  1. Enable USE_MKL2017=1 in make/config.mk
    1.1 USE_BLAS should be atlas by default
    1.2 if need USE_BLAS to be mkl, please  Navigate here - https://registrationcenter.intel.com/en/forms/?productid=2558&licensetype=2 to do a full MKL installation
  2. Run 'make -jX'
    2.1 Makefile will execute "prepare_mkl.sh" to download the mkl under root folder.e.g. <MXNET ROOTDIR> /mklml_lnx_2017.0.0.20160801
    2.2 if download failed because of proxy setting, please do it manually before make
    2.2.1 wget https://github.com/intel/caffe/releases/download/self_containted_MKLGOLD/mklml_lnx_2017.0.0.20160801.tgz
    2.2.2 tar zxvf mklml_lnx_2017.0.0.20160801.tgz

  3. Navigate into the python directory
  4. Run 'sudo python setup.py install'
  5. Before excute python scipt, need to set LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<MXNET ROOTDIR>/mklml_lnx_2017.0.0.20160801/lib
```

