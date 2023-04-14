import cv2, pandas as pd
import os, numpy as np
from torch.utils.data import Dataset
from utils import resize_and_center_fundus
from tqdm import tqdm
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
        
        self.images = []
        for image_path in tqdm(self.image_paths):
            if os.path.exists(f"data/EyePACS/{image_path.split('/')[-1]}"):
                image = cv2.imread(f"data/EyePACS/{image_path.split('/')[-1]}", -1)
                self.images.append(image)
            else:
                image = self.preprocess(image_path)
                cv2.imwrite(
                    f"data/EyePACS/{image_path.split('/')[-1]}",
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                )
                self.images.append(image)

        self.images = np.stack(self.images)
        print(self.images.shape)

        self.labels = pd.read_csv(root + "trainLabels.csv")["level"].values
        self.labels = self.labels[: len(self.image_paths)]
        print("Loaded {} images.".format(len(self.image_paths)))

    def preprocess(self, image_path):
        image = cv2.imread(image_path, -1)
        processed = resize_and_center_fundus(image, diameter=224)
        if processed is None:
            raise ValueError("Could not preprocess {}".format(image_path))
        else:
            return processed

    def distort_label(self, label):
        row = self.markov_matrix[label]
        return np.random.choice(5, p=row)
    
    def numpy_to_pil(self, image):
        image = image.astype(np.uint8)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return Image.fromarray(image)

    def __getitem__(self, index):
        image = self.images[index]
        image = self.numpy_to_pil(image)
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
