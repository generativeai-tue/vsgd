"""VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
"""

import torch.nn as nn

from model.classifier import ClassifierWrapper

# adapted from https://github.com/locuslab/convmixer-cifar10/blob/main/train.py


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(ClassifierWrapper):
    def __init__(
        self,
        dim,
        depth,
        kernel_size=5,
        patch_size=2,
        num_classes=100,
        loss_fn=nn.CrossEntropyLoss(),
        **kwargs
    ):
        bb = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                dim, dim, kernel_size, groups=dim, padding="same"
                            ),
                            nn.GELU(),
                            nn.BatchNorm2d(dim),
                        )
                    ),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )
                for i in range(depth)
            ],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, num_classes)
        )
        super().__init__(backbone=bb, loss_fn=loss_fn)

    def forward(self, batch):
        x = self.backbone(batch[0])
        return x
