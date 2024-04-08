"""VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
"""
import math

import torch.nn as nn
import torch.nn.functional as F

from model.classifier import ClassifierWrapper

# adapted from https://github.com/alecwangcq/KFAC-Pytorch/blob/master/models/cifar/vgg.py

__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]


cfg = {
    # vgg11
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    #  vgg13:
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    #  vgg16:
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    #  vgg19:
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(ClassifierWrapper):
    def __init__(
        self,
        cfg_id,
        batch_norm=False,
        num_classes=1000,
        loss_fn=nn.CrossEntropyLoss(),
        **kwargs
    ):
        super().__init__(
            backbone=VGG.make_layers(cfg[cfg_id], batch_norm=batch_norm),
            loss_fn=loss_fn,
        )
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, batch):
        x = self.backbone(batch[0])
        x = F.avg_pool2d(x, kernel_size=x.shape[-1], stride=1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    @staticmethod
    def make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
