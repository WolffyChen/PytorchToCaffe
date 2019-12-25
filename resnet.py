import torch
import torch.nn as nn

from collections import OrderedDict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']


model_paths = {
    'resnet18': 'resnet18.pth',
    'resnet34': 'resnet34.pth',
    'resnet50': 'resnet50.pth',
    'resnet101': 'resnet101.pth',
    'resnet152': 'resnet152.pth',
    'resnext50_32x4d': 'resnext50_32x4d.pth',
    'resnext101_32x8d': 'resnext101_32x8d.pth',
    'wide_resnet50_2': 'wide_resnet50_2.pth',
    'wide_resnet101_2': 'wide_resnet101_2.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.left = nn.Sequential(OrderedDict([
            ('conv1', conv3x3(inplanes, planes, stride)),
            ('bn1', norm_layer(planes)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv2', conv3x3(planes, planes)),
            ('bn2', norm_layer(planes))
        ]))

        self.shortcut = nn.Sequential() if downsample is None else downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.left = nn.Sequential(OrderedDict([
            ('conv1', conv1x1(inplanes, width)),
            ('bn1', norm_layer(width)),
            ('conv2', conv3x3(width, width, stride, groups, dilation)),
            ('bn2', norm_layer(width)),
            ('conv3', conv1x1(width, planes * self.expansion)),
            ('bn3', norm_layer(planes * self.expansion)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.shortcut = nn.Sequential() if downsample is None else downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', norm_layer(self.inplanes)),
            ('relu', nn.ReLU(inplace=True))
        ]))
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.left.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.left.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.layer0(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

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


def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.load(model_paths[arch])
        model.load_state_dict(state_dict, strict=False)

    return model


def resnet18(pretrained=False, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, **kwargs)

    return model


def resnet34(pretrained=False, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, **kwargs)

    return model


def resnet50(pretrained=False, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)

    return model


def resnet101(pretrained=False, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, **kwargs)

    return model


def resnet152(pretrained=False, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, **kwargs)

    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4

    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)


def resnext101_32x8d(pretrained=False, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8

    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, **kwargs)


def wide_resnet50_2(pretrained=False, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['width_per_group'] = 64 * 2

    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)


def wide_resnet101_2(pretrained=False, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['width_per_group'] = 64 * 2

    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, **kwargs)

