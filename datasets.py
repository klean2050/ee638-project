import cv2, pandas as pd, numpy as np
import os, matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from utils import resize_and_center_fundus
from tqdm import tqdm
from PIL import Image


class EyePACS(Dataset):
    def __init__(self, root, split, contrastive, transform=None, label_transform=False):
        self.split = split
        self.contrastive = contrastive
        self.transform = (
            transform
            if split == "train"
            else transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        )

        self.label_transform = label_transform if split == "train" else False
        if self.label_transform:
            # self.markov_matrix = self.configure_matrix()
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

        # load and align labels with images
        self.names = [i.split("/")[-1] for i in self.image_paths]
        temp = pd.read_csv(root + "trainLabels.csv")

        self.images, self.labels = [], []
        for image_path in tqdm(self.names):
            if not os.path.exists(f"data/EyePACS/{image_path}"):
                image = self.preprocess(f"{root}set_0/{image_path}")
                if image.shape[-1] != 3:
                    continue
                cv2.imwrite(f"data/EyePACS/{image_path}", image)

            self.images.append(f"data/EyePACS/{image_path}")
            self.labels.append(
                temp[temp["image"] == image_path.strip(".jpeg")]["level"].values[0]
            )

        print("Loaded {} images.".format(len(self.images)))

    def preprocess(self, image_path):
        image = cv2.imread(image_path, -1)
        processed = resize_and_center_fundus(image, diameter=512)
        if processed is None:
            print("NOTE: Could not preprocess {}".format(image_path))
            return np.zeros((512, 512, 2))
        else:
            return processed

    def configure_matrix(self):
        table = []
        for i in range(5):
            row = []
            for j in range(5):
                value = abs(i - j)
                row.append(5 - value)
            table.append(row)
        table = np.array(table)
        table[np.diag_indices(5)] = 0
        table = table / table.sum(axis=1, keepdims=True) / 2
        table[np.diag_indices(5)] = 0.5
        return table

    def distort_label(self, label):
        row = self.markov_matrix[label]
        return np.random.choice(5, p=row)

    def __getitem__(self, index):
        image = plt.imread(self.images[index])[..., :3]
        image = Image.fromarray(np.uint8(image))
        label = self.labels[index]

        if self.label_transform and self.split == "train":
            label = self.distort_label(label)

        if self.contrastive:
            x1 = self.transform(image.copy())
            x2 = self.transform(image.copy())
            return x1, x2, label
        else:
            return self.transform(image), label

    def __len__(self):
        return len(self.images)


class VinDR(Dataset):
    def __init__(self, root, split, contrastive, transform=None, label_transform=False):
        self.root = root
        self.split = split
        self.contrastive = contrastive
        self.transform = (
            transform
            if split == "train"
            else transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        )

        self.label_transform = label_transform if split == "train" else False
        if self.label_transform:
            self.markov_matrix = np.random.rand(8, 8)
            for i in range(8):
                self.markov_matrix[i, i] = 0.5
                self.markov_matrix[i][i != np.arange(8)] /= 2 * np.sum(
                    self.markov_matrix[i][i != np.arange(8)]
                )

        if split == "train":
            self.image_paths = [
                os.path.join(root, "train_images/", i)
                for i in os.listdir(root + "train_images/")
            ]
            self.image_paths = self.image_paths[: int(0.8 * len(self.image_paths))]
        elif split == "valid":
            self.image_paths = [
                os.path.join(root, "train_images/", i)
                for i in os.listdir(root + "train_images/")
            ]
            self.image_paths = self.image_paths[int(0.8 * len(self.image_paths)) :]
        else:
            self.image_paths = [
                os.path.join(root, "test_images/", i)
                for i in os.listdir(root + "test_images/")
            ]
        self.image_paths = sorted(self.image_paths)

        # Load labels
        self.labels = self.configure_label("train" if split == "valid" else split)
        label_set = sorted(set(self.labels["lesion_type"].values))
        mapping = {label: i for i, label in enumerate(label_set)}
        self.labels = self.labels["lesion_type"].map(mapping).values

        print("Loaded {} images.".format(len(self.labels)))

    def configure_label(self, split):
        annotations = pd.read_csv(self.root + f"/annotations/{split}.csv")
        self.labels = [
            annotations.iloc[idx]
            for idx, row in enumerate(annotations.iterrows())
            if "{}_images/{}.jpg".format(self.root + split, row[1]["image_id"])
            in self.image_paths
        ]
        self.labels = (
            pd.DataFrame(self.labels)
            .drop_duplicates(subset=["image_id"])
            .sort_values(by=["image_id"])
        )
        return self.labels[["image_id", "lesion_type"]]

    def configure_matrix(self):
        raise NotImplementedError

    def distort_label(self, label):
        row = self.markov_matrix[label]
        return np.random.choice(8, p=row)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        label = self.labels[index]
        if self.label_transform and self.split == "train":
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

    dataset = "eyepacs"
    transform = Compose(
        [
            RandomResizedCrop((224, 224)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ColorJitter(brightness=0.2, saturation=0.2, hue=0.2),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if dataset == "eyepacs":
        data = EyePACS(
            root="/data/avramidi/large_fundus/",
            split="train",
            contrastive=False,
            transform=transform,
            label_transform=True,
        )
    else:
        data = VinDR(
            root="/data/avramidi/tiny_vindr/",
            split="valid",
            contrastive=False,
            transform=transform,
            label_transform=True,
        )
    sample, label = data[0]
    print(sample.shape, label)
