"""
    BottleNeck block만 만들었읍니다..
"""

import torch
import torch.nn as nn


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, plane, kernel_size, stride, downsample):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, plane, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(plane),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(plane, plane, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(plane),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(plane, plane * self.expansion, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(plane * self.expansion),
        )

    def forward(self, x):
        shortcut = x
        if self.downsample:
            shortcut = self.downsample(shortcut)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return nn.ReLU()(x + shortcut)


class FPN(nn.Module):
    def __init__(self, block, repeat):
        super().__init__()
        self.in_planes = 64
        self.base = nn.Sequential(
            nn.Conv2d(3, self.in_planes, 7, 2, padding=1),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(3, 2)

        self.bottom_up1 = self._make_layers(block, 64, 1, repeat[0])
        self.bottom_up2 = self._make_layers(block, 128, 2, repeat[1])
        self.bottom_up3 = self._make_layers(block, 256, 2, repeat[2])
        self.bottom_up4 = self._make_layers(block, 512, 2, repeat[3])

        self.top_layer = nn.Conv2d(2048, 256, 1, 1, 0)

        self.lateral_1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.lateral_2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.lateral_3 = nn.Conv2d(256, 256, 1, 1, 0)

        self.smooth_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.smooth_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.smooth_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.base(x)
        x = self.maxpool(x)

        c1 = self.bottom_up1(x)
        c2 = self.bottom_up2(c1)
        c3 = self.bottom_up3(c2)
        c4 = self.bottom_up4(c3)

        p4 = self.top_layer(c4)

        p3 = self.upscale_add(p4, self.lateral_1(c3))
        p2 = self.upscale_add(p3, self.lateral_2(c2))
        p1 = self.upscale_add(p2, self.lateral_3(c1))

        p3 = self.smooth_1(p3)
        p2 = self.smooth_2(p2)
        p1 = self.smooth_3(p1)

        return p1, p2, p3, p4

    def _make_layers(self, block, plane, stride, repeat):
        layers = []
        strides = [stride] + [1] * (repeat - 1)

        for s in strides:
            downsample = None
            if s != 1 or self.in_planes != block.expansion * plane:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_planes, plane * block.expansion, kernel_size=1, padding=0, stride=s),
                    nn.BatchNorm2d(plane * block.expansion)
                )
            layers.append(block(self.in_planes, plane, 3, s, downsample))
            self.in_planes = plane * block.expansion

        return nn.Sequential(*layers)

    def upscale_add(self, x, y):
        _, _, H, W = y.size()
        return nn.functional.upsample(x, size=(H, W), mode='bilinear') + y




def FPN101():
    return FPN(BottleNeck, [2,4,23,3])
    # return FPN(Bottleneck, [2,2,2,2])


def test():
    net = FPN101()
    fms = net(torch.tensor(torch.randn(1,3,600,900)))
    for fm in fms:
        print(fm.size())

test()