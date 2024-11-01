# This repository is based on [PytorchToCaffe](https://github.com/xxradon/PytorchToCaffe).

## How to use
- Install Pytorch and Caffe.
    - Create a conda virtual environment and activate it.
    - Install PyTorch following the [official instructions](https://pytorch.org).
    - Install Caffe by `conda install -c defaults caffe`.
    - Install OpenCV by `pip install opencv-python`.
- Run `python resnet_run.py` to convert the resnet model.
- Run `python resnet_verify.py` to verify the precison.

