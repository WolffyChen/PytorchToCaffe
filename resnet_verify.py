import torch
import torchvision
import resnet
from torch.autograd import Variable
import cv2
import caffe
import numpy as np


if __name__ == '__main__':
    name = 'resnet'
    model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load('{}.pth'.format(name), map_location='cpu'))
    model.eval()

    img = cv2.imread('erha.jpg')
    img = cv2.resize(img, (112, 112))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    img = torch.from_numpy(img).unsqueeze(0)

    pred = model(img)
    for i in range(len(pred)):
        print('{} {} {}'.format(pred[i].shape, torch.max(pred[i]), torch.min(pred[i])))
        print(pred[i][:])

    net = caffe.Net('{}.prototxt'.format(name), '{}.caffemodel'.format(name), caffe.TEST)
    net.blobs['data'].reshape(1, 3, 112, 112)
    net.blobs['data'].data[...] = img.numpy()
    net.forward()
    caffe_pred = []
    for i in range(len(net.outputs)):
        caffe_pred.append(net.blobs[net.outputs[i]].data)
        print(net.blobs[net.outputs[i]].data.shape)
        print(net.blobs[net.outputs[i]].data[:])

    for i in range(len(pred)):
        print('output%d have %10.10f%% relative error'%(i, np.linalg.norm(caffe_pred[i] - pred[i].detach().numpy()) / np.linalg.norm(pred[i].detach().numpy()) * 100))

