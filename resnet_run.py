import sys

sys.path.insert(0, '..')
import torch
import torchvision
import resnet
from torch.autograd import Variable
import pytorch_to_caffe

if __name__ == '__main__':
    name = 'resnet'
    model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2])
    #model.load_state_dict(torch.load('{}.pt'.format(name), map_location='cpu'))
    torch.save(model.state_dict(), '{}.pth'.format(name))
    model.fuse()
    model.eval()
    def apply_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()
    model.apply(apply_dropout)
    input = torch.ones([1, 3, 112, 112])
    pytorch_to_caffe.trans_net(model, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))

