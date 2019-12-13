import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.left = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(planes)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(planes))
        ]))

        self.shortcut = nn.Sequential() if downsample is None else downsample
        self.stride = stride

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.left = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)),
            ('bn1', nn.BatchNorm2d(planes)),
            ('conv2', nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(planes)),
            ('conv3', nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)),
            ('bn3', nn.BatchNorm2d(planes * self.expansion)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.shortcut = nn.Sequential() if downsample is None else downsample
        self.stride = stride

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU(inplace=True))
        ]))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, Bottleneck):
                print("INIT BN ", m)
                nn.init.constant_(m.left.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        for na, a in list(self.named_children()):
            if isinstance(a, nn.Sequential):
                if 'layer0' == na:
                    for i, b in enumerate(a):
                        if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                            # fuse this bn layer with the previous conv2d layer
                            conv = a[i-1]
                            fused = fuse_conv_and_bn(conv, b)
                            a = nn.Sequential(fused, *list(a.children())[i+1:])
                            #print(a)
                            break
                    self.layer0 = a
                elif 'layer' in na:
                    for i, b in enumerate(a):
                        left_list = []
                        for j, c in enumerate(b.left):
                            if isinstance(c, nn.modules.batchnorm.BatchNorm2d):
                                # fuse this bn layer with the previous conv2d layer
                                conv = a[i].left[j-1]
                                fused = fuse_conv_and_bn(conv, c)
                                left_list[-1] = fused
                            else:
                                left_list.append(c)
                        a[i].left = nn.Sequential(*left_list)
                        #print(a[i].left)

                        for j, c in enumerate(b.shortcut):
                            if isinstance(c, nn.modules.batchnorm.BatchNorm2d):
                                # fuse this bn layer with the previous conv2d layer
                                conv = a[i].shortcut[j-1]
                                fused = fuse_conv_and_bn(conv, c)
                                a[i].shortcut = nn.Sequential(fused, *list(b.shortcut.children())[j+1:])
                                #print(a[i].shortcut)
                                break
                else:
                    print('skip {}...'.format(na))


def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = torch.nn.Conv2d(conv.in_channels,
                                    conv.out_channels,
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    bias=True)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(b_conv + b_bn)

        return fusedconv


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict('resnet18.pth')

    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict('resnet34.pth')

    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict('resnet50.pth')

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict('resnet101.pth')

    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict('resnet152.pth')

    return model

