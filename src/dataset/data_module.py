from itertools import permutations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


class ToTensor:
    def __call__(self, x):
        x = torch.FloatTensor(np.asarray(x, dtype=np.float32)).permute(2, 0, 1)
        return x


class Random90Rotation:
    def __call__(self, x):
        k = torch.ceil(3.0 * torch.rand(1)).long()
        u = torch.rand(1)
        if u < 0.5:
            x = x.rotate(90 * k)
        return x


class ChannelSwap:
    def __call__(self, x):
        permutation = list(permutations(range(3), 3))[np.random.randint(0, 5)]
        u = torch.rand(1)
        if u < 0.5:
            x = np.array(x)[..., permutation]
            x = Image.fromarray(x)
        return x


class DataModule:
    def __init__(
        self,
        batch_size,
        test_batch_size,
        root="data/",
    ):
        self.__dict__.update(locals())
        self.transforms = transforms.Compose(
            [
                ToTensor(),
            ]
        )
        self.test_transforms = transforms.Compose(
            [
                ToTensor(),
            ]
        )
        self.prepare_data()

    def prepare_data(self) -> None:
        """
        Download the data. Do preprocessing if necessary.
        :return:
        """
        raise NotImplementedError

    def setup(self) -> None:
        """
        Create self.train and self.val, self.test dataset
        :return: None
        """
        raise NotImplementedError

    def train_dataloader(self):
        params = {
            "pin_memory": True,
            "drop_last": True,
            "shuffle": True,
            "num_workers": 1,
        }
        train_loader = DataLoader(self.train, self.batch_size, **params)
        while True:
            yield from train_loader

    def val_dataloader(self):
        params = {
            "pin_memory": True,
            "drop_last": True,
            "shuffle": True,
            "num_workers": 1,
        }
        val_loader = DataLoader(self.val, self.test_batch_size, **params)
        while True:
            yield from val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test,
            self.test_batch_size,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        return test_loader
