import glob
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.data_module import DataModule, ToTensor


class TinyImagenet(DataModule):
    def __init__(
        self,
        batch_size,
        test_batch_size,
        root,
        use_augmentations,
    ):
        super().__init__(
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            root=root,
        )
        self.__dict__.update(locals())
        if use_augmentations:
            self.transforms = transforms.Compose(
                [
                    transforms.RandomCrop(size=64, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(
                        degrees=45, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            self.test_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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

    def prepare_data(self) -> None:
        pass

    def setup(self):
        self.train = TinyImageNet(self.root, split="train", transform=self.transforms)
        self.val = TinyImageNet(self.root, split="val", transform=self.test_transforms)
        self.test = TinyImageNet(self.root, split="val", transform=self.test_transforms)


EXTENSION = "JPEG"
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = "wnids.txt"
VAL_ANNOTATION_FILE = "val_annotations.txt"


class TinyImageNet(Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Dataset code adapted from https://github.com/leemengtw/tiny-imagenet/blob/master/TinyImageNet.py

    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        in_memory=False,
    ):
        self.root = os.path.join(os.path.expanduser(root), "tiny-imagenet-200")
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(self.root, self.split)
        self.image_paths = sorted(
            glob.iglob(
                os.path.join(self.split_dir, "**", "*.%s" % EXTENSION), recursive=True
            )
        )
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing

        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), "r") as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == "train":
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels["%s_%d.%s" % (label_text, cnt, EXTENSION)] = i
        elif self.split == "val":
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), "r") as fp:
                for line in fp.readlines():
                    terms = line.split("\t")
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)
        if self.split == "test":
            return img
        else:
            # file_name = file_path.split('/')[-1]
            return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = self.split
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

    def read_image(self, path):
        # img = imageio.imread(path, pilmode='RGB')
        img = Image.open(path)
        img = img.convert("RGB")
        return self.transform(img) if self.transform else img
