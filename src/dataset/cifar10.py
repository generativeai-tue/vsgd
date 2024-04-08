import os

from torch.utils.data import random_split
from torchvision import datasets, transforms

from dataset.data_module import DataModule, ToTensor


class Cifar10(DataModule):
    def __init__(
        self,
        batch_size,
        test_batch_size,
        root,
        use_augmentations,
    ):
        super(Cifar10, self).__init__(
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            root=root,
        )
        self.__dict__.update(locals())
        if use_augmentations:
            self.transforms = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            self.test_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

        else:
            self.transforms = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    ToTensor(),
                ]
            )
            self.test_transforms = transforms.Compose([ToTensor()])
        self.prepare_data()

    def prepare_data(self):
        datasets.CIFAR10(self.root, train=True, download=True)
        datasets.CIFAR10(self.root, train=False, download=True)

    def setup(self):
        cifar_full = datasets.CIFAR10(self.root, train=True, transform=self.transforms)
        cifar_full.processed_folder = os.path.join(self.root, cifar_full.base_folder)
        N = len(cifar_full)
        self.train = cifar_full
        self.train, self.val = random_split(cifar_full, [N - 256, 256])
        self.test = datasets.CIFAR10(
            self.root, train=False, transform=self.test_transforms
        )
        self.test.processed_folder = os.path.join(self.root, self.test.base_folder)
