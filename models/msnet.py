import torch.nn as nn
import torch
import math
import pdb
import torch.nn.functional as F
import numpy as np



class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, dilation=1):
        super(BasicConv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation=dilation, groups=groups, bias=False)


    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck,
                 bnWidth):
        """
        a basic conv in EANet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottlenet or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.BatchNorm2d(int(nIn)))
            layer.append(nn.ReLU(True))
            layer.append(nn.Conv2d(
                  nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))

        if type == 'normal':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=1, padding=1, bias=False))
        elif type == 'down':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=2, padding=1, bias=False))
        else:
            raise ValueError

        self.net = nn.Sequential(*layer)

    def forward(self, x):

        return self.net(x)

class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut1, nOut2, bottleneck, bnWidth1, bnWidth2, args):
        super(ConvDownNormal, self).__init__()
        
        self.conv_down = ConvBN(nIn1 + nOut1 // 3, math.floor(nOut2*args.compress_factor), 'down',
                                bottleneck, bnWidth1)
        self.conv_up = BasicConv(nIn2, nOut1 // 3, kernel_size=1, stride=1, padding=0, groups=1)  #math.floor(nOut1*compress_factor)
        self.conv_normal = ConvBN(nIn2, nOut2-math.floor(nOut2*args.compress_factor), 'normal', 
                                  bottleneck, bnWidth2)


    def forward(self, x):
        x_dense = x[1]
        _,_,h,w = x[0].size()
        x0 = torch.cat([x[0], -F.interpolate(self.conv_up(x[1]), size=(h,w), mode = 'bilinear', align_corners=True)], dim=1)
        res = [self.conv_down(x0), 
               self.conv_normal(x[1])]

        x = torch.cat(res, dim=1)

        return torch.cat([x_dense, x], dim=1)

class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        super(ConvNormal, self).__init__()

        self.conv_normal = ConvBN(nIn, nOut, 'normal',
                                   bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        x_dense = x[0]
        x = self.conv_normal(x[0])

        return torch.cat([x_dense, x], dim=1)

class Transition(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Transition, self).__init__()

        in_channels = in_channels + mid_channels
        self.reduce_channels = (in_channels - out_channels) // 2
        self.conv1_1x1 = BasicConv(in_channels, in_channels-self.reduce_channels, kernel_size=1, stride=1)
        self.conv2_3x3 = BasicConv(in_channels-self.reduce_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=1)
        if mid_channels != 0:
            self.conv_h = BasicConv(mid_channels, mid_channels ,kernel_size=3, stride=1, padding=1, groups=mid_channels)
        self.conv_s = nn.Sequential(
                    BasicConv(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels),
                    BasicConv(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
                    )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1)
        self.fc2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)

    def forward(self, s, x, h):

        if h is not None:
            out_h = self.conv_h(h)
            out_h += h
            x = torch.cat([x, out_h], dim=1)
        x = self.conv1_1x1(x)
        x = self.conv2_3x3(x)
        if s is not None:
            out_s = self.conv_s(s)
            x += out_s
        x_shortcut = x
        x_shortcut_1 = self.gap(x_shortcut)
        x_shortcut_1 = F.relu(self.fc1(x_shortcut_1))
        x_shortcut_1 = torch.sigmoid(self.fc2(x_shortcut_1))
        x = x_shortcut + x_shortcut * x_shortcut_1

        return x

class MSNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args):
        super(MSNFirstLayer, self).__init__()
        self.nScales = args.nScales
        self.grFactor = args.grFactor
        self.layers = nn.ModuleList()

        if args.data.startswith('cifar'):
            self.layers.append(nn.Conv2d(nIn, nOut * args.grFactor[0],
                                         kernel_size=3, stride=1, padding=1))
        elif args.data == 'ImageNet':
            conv = nn.Sequential(
                    nn.Conv2d(nIn, args.growthRate, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(args.growthRate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(args.growthRate, args.growthRate, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(args.growthRate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(args.growthRate, nOut * args.grFactor[0], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(nOut * args.grFactor[0]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, 2, 1))
            self.layers.append(conv)

        nIn = nOut * args.grFactor[0]

        for i in range(1, args.nScales):
            self.layers.append(BasicConv(nIn, nOut * args.grFactor[i],
                                         kernel_size=3, stride=2, padding=1))
            nIn = nOut * args.grFactor[i]

    def forward(self, x):
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)

        return res

class MSNLayer(nn.Module):
    def __init__(self, nIn, nOut, args, inScales=None, outScales=None):
        super(MSNLayer, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.inScales = inScales if inScales is not None else args.nScales
        self.outScales = outScales if outScales is not None else args.nScales

        self.layers = nn.ModuleList()

        self.layers.append(ConvNormal(nIn * args.grFactor[0],
                                      nOut * args.grFactor[0],
                                      args.bottleneck,
                                      args.grFactor[0]))

        for i in range(1, self.outScales):
            nIn1 = nIn * args.grFactor[i - 1]
            nIn2 = nIn * args.grFactor[i]
            nOut1 = nOut * args.grFactor[i - 1]
            nOut2 = nOut * args.grFactor[i]
            self.layers.append(ConvDownNormal(nIn1, nIn2, nOut1, nOut2, args.bottleneck,
                                              args.grFactor[i - 1],
                                              args.grFactor[i], args))

    def forward(self, x):
        
        inp = [[x[0]]]
        for i in range(1, self.outScales):
            inp.append([x[i-1], x[i]])

        res = []
        for i in range(self.outScales):
            res.append(self.layers[i](inp[i]))

        return res


class ParallelModule(nn.Module):
    """
    This module is similar to luatorch's Parallel Table
    input: N tensor
    network: N module
    output: N tensor
    """
    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        _x = self.m[0](None, x[0], None)
        res.append(_x)
        for i in range(1, len(x)-1):
            _x = self.m[i](x[i-1], x[i], _x)
            res.append(_x)
            
        return res


class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.m = m
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(channel, num_classes)

    def forward(self, x):
        res = self.m(x[-1])
        f = res
        res = self.global_pool(res)
        res = res.view(res.size(0), -1)
        return self.linear(res), f

class MSNet(nn.Module):
    def __init__(self, args):
        super(MSNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.nBlocks = args.nBlocks
        self.steps = args.step
        self.args = args
        
        if self.nBlocks != len(self.steps):
            raise ValueError('NUM_STEPS({}) <> NUM_BLOCKS({})'.format(len(self.steps), self.nBlocks))
        n_layers_all = sum(self.steps)

        print("building network of steps: ")
        print(self.steps, n_layers_all)

        nIn = args.nChannels
        inScales = args.nScales
        outScales = args.nScales
        for i in range(self.nBlocks):
            inScales = outScales
            if i in args.transFactor:
                outScales -= 1
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m, nIn = \
                self._build_block(nIn, args, self.steps[i],
                                  inScales, outScales)
            self.blocks.append(m)

            if args.data.startswith('cifar100'):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * args.grFactor[outScales-1], 100))
            elif args.data.startswith('cifar10'):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * args.grFactor[outScales-1], 10))
            elif args.data == 'ImageNet':
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * args.grFactor[outScales-1], 1000))
            else:
                raise NotImplementedError

        for m in self.blocks:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

        for m in self.classifier:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_block(self, nIn, args, step, inScales, outScales):

        layers = [MSNFirstLayer(3, nIn, args)] \
            if nIn == args.nChannels else []
        for i in range(step):
            if args.prune == 'max' and inScales > outScales and \
                    args.reduction > 0:
                layers.append(
                    self._build_transition(nIn, math.floor(1.0 * args.reduction * nIn),
                                           outScales, args))
                _t = nIn
                nIn = math.floor(1.0 * args.reduction * nIn)
                print('|\t\tTransition layer inserted! (max) inChannels {} outChannels {}\t|'.format(_t, math.floor(1.0 * args.reduction * _t)))
            elif args.prune == 'min' and args.reduction > 0 and \
                    ((n_layer_curr == math.floor(1.0 * n_layer_all / 3)) or
                     n_layer_curr == math.floor(2.0 * n_layer_all / 3)):
                
                layers.append(self._build_transition(nIn, math.floor(1.0 * args.reduction * nIn),
                                                     outScales, args))

                nIn = math.floor(1.0 * args.reduction * nIn)
                print('|\t\tTransition layer inserted! (min)\t|')

            offset = args.nScales - outScales
            growthRate = args.growthRate * args.grFactor[offset]

            layers.append(MSNLayer(nIn, growthRate, args, inScales, outScales))
            print('|\t\tinScales {} outScales {} inChannels {} outChannels {}\t\t|'.format(inScales, outScales, nIn, growthRate))
            
            inScales = outScales
            nIn += growthRate

            print("")

        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, outScales, args):
        net = []
        net.append(Transition(nIn * args.grFactor[0], 0, nOut * args.grFactor[0]))
        for i in range(1, outScales):
            net.append(Transition(nIn * args.grFactor[i], nOut * args.grFactor[i-1],
                                  nOut * args.grFactor[i]))

        return ParallelModule(net)

    def _build_classifier_cifar(self, nIn, num_classes):
 
        if nIn >= 256:
            interChannels1, interChannels2 = 128, 128
        elif nIn >= 128:
            interChannels1, interChannels2 = 64, 64
        else:
            interChannels1, interChannels2 = 32, 32
        conv = nn.Sequential(
            BasicConv(nIn, interChannels1, kernel_size=3, stride=2, padding=1),
            BasicConv(interChannels1, interChannels2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(interChannels2),
            nn.ReLU(inplace=True)
            #nn.AdaptiveAvgPool2d((1, 1))
        )
        return ClassifierModule(conv, interChannels2, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):

        conv = nn.Sequential(
            BasicConv(nIn, nIn, kernel_size=3, stride=2, padding=1),
            BasicConv(nIn, nIn, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nIn),
            nn.ReLU(inplace=True)
            #nn.AdaptiveAvgPool2d((1, 1))
        )
        return ClassifierModule(conv, nIn, num_classes)

    def forward(self, x):
        res = []
        f =[]
        for i in range(self.nBlocks):
            x = self.blocks[i](x)
            logit, feat = self.classifier[i](x)
            res.append(logit)
            f.append(feat)
        return res, f

if __name__ == '__main__':
    from args import arg_parser
    from op_counter import measure_model
    #from op import measure_model

