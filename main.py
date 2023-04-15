import os, torch, pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms import *

from datasets import EyePACS, VinDR
from models import GlobalModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = "/data/avramidi/tiny_vindr/"  # large_fundus or tiny_vindr
hparams = {
    "dataset": "VinDR",
    "contrastive": False,
    "lr": 1e-3,
    "batch_size": 64,
    "max_epochs": 100,
    "weight_decay": 1e-6,
    "distort": False,
}
data_transform = Compose(
    [
        RandomResizedCrop((224, 224)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
logger = TensorBoardLogger(
    save_dir="logging/",
    name=hparams["dataset"],
)


def get_eyepacs(contrastive, label_transform):
    train_dataset = EyePACS(
        root=root,
        split="train",
        contrastive=contrastive,
        transform=data_transform,
        label_transform=label_transform,
    )
    valid_dataset = EyePACS(
        root=root,
        split="valid",
        contrastive=contrastive,
        transform=data_transform,
        label_transform=label_transform,
    )
    return train_dataset, valid_dataset, None


def get_vindr(contrastive, label_transform):
    train_dataset = VinDR(
        root=root,
        split="train",
        contrastive=contrastive,
        transform=data_transform,
        label_transform=label_transform,
    )
    valid_dataset = VinDR(
        root=root,
        split="valid",
        contrastive=contrastive,
        transform=data_transform,
        label_transform=label_transform,
    )
    test_dataset = VinDR(
        root=root,
        split="test",
        contrastive=contrastive,
        transform=data_transform,
        label_transform=label_transform,
    )
    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":

    train_dataset, valid_dataset, test_dataset = (
        get_vindr(
            contrastive=hparams["contrastive"], label_transform=hparams["distort"]
        )
        if hparams["dataset"] == "VinDR"
        else get_eyepacs(
            contrastive=hparams["contrastive"], label_transform=hparams["distort"]
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=hparams["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )
        if test_dataset is not None
        else None
    )

    model = GlobalModel(hparams).to(device)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=hparams["max_epochs"],
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
        sync_batchnorm=True,
        strategy="ddp_find_unused_parameters_false",
        accelerator="gpu",
        devices="auto",
        precision=16,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    if test_loader is not None:
        trainer.test(model, test_dataloaders=test_loader)
