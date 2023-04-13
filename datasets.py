import os, pandas as pd, numpy as np
from torch.utils.data import Dataset
from PIL import Image


class EyePACS(Dataset):
    def __init__(self, root, split, contrastive, transform=None, label_transform=False):
        self.contrastive = contrastive
        self.transform = transform

        self.label_transform = label_transform if split == "train" else False
        if self.label_transform:
            self.markov_matrix = np.random.rand(5, 5)
            for i in range(5):
                self.markov_matrix[i, i] = 0.5
                self.markov_matrix[i][i != np.arange(5)] /= 2 * np.sum(
                    self.markov_matrix[i][i != np.arange(5)]
                )

        self.image_paths = [
            os.path.join(root, "set_0", i) for i in os.listdir(root + "set_0/")
        ]
        if split == "train":
            self.image_paths = self.image_paths[: int(0.8 * len(self.image_paths))]
        elif split == "valid":
            self.image_paths = self.image_paths[int(0.8 * len(self.image_paths)) :]
        else:
            raise ValueError("Not implemented split")

        self.labels = pd.read_csv(root + "trainLabels.csv")["level"].values
        self.labels = self.labels[: len(self.image_paths)]
        print("Loaded {} images.".format(len(self.image_paths)))

    def distort_label(self, label):
        row = self.markov_matrix[label]
        return np.random.choice(5, p=row)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        label = self.labels[index]
        if self.label_transform:
            label = self.distort_label(label)

        if self.contrastive:
            x1 = self.transform(image.copy())
            x2 = self.transform(image.copy())
            return x1, x2, label
        else:
            return self.transform(image), label

    def __len__(self):
        return len(self.image_paths)


class VinDR(Dataset):
    def __init__(self, root, split, contrastive, transform=None, label_transform=False):
        self.contrastive = contrastive
        self.transform = transform

        self.label_transform = label_transform if split == "train" else False
        if self.label_transform:
            self.markov_matrix = np.random.rand(5, 5)
            for i in range(5):
                self.markov_matrix[i, i] = 0.5
                self.markov_matrix[i][i != np.arange(5)] /= 2 * np.sum(
                    self.markov_matrix[i][i != np.arange(5)]
                )

        self.image_paths = [
            os.path.join(root, "set_0", i) for i in os.listdir(root + "set_0/")
        ]
        if split == "train":
            self.image_paths = self.image_paths[: int(0.8 * len(self.image_paths))]
        elif split == "valid":
            self.image_paths = self.image_paths[int(0.8 * len(self.image_paths)) :]
        else:
            raise ValueError("Not implemented split")

        self.labels = pd.read_csv(root + "trainLabels.csv")["level"].values
        self.labels = self.labels[: len(self.image_paths)]
        print("Loaded {} images.".format(len(self.image_paths)))

    def distort_label(self, label):
        row = self.markov_matrix[label]
        new_label = np.random.choice(5, p=row)
        return row[new_label]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        label = self.labels[index]
        if self.label_transform:
            label = self.distort_label(label)

        if self.contrastive:
            x1 = self.transform(image.copy())
            x2 = self.transform(image.copy())
            return x1, x2, label
        else:
            return self.transform(image), label

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    from torchvision.transforms import *

    root = "/data/avramidi/large_fundus/"
    transform = Compose(
        [
            RandomResizedCrop(224, scale=(0.2, 1.0)),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    eyepacs_set = EyePACS(
        root=root,
        split="train",
        contrastive=False,
        transform=transform,
        label_transform=True,
    )
    sample, label = eyepacs_set[0]
    print(sample.shape, label)
