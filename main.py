import os, torch, pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms import *

from datasets import EyePACS, VinDR
from models import EyePACS_Model, VinDR_Model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = "/data/avramidi/large_fundus/"
hparams = {
    "dataset": "EyePACS",
    "contrastive": False,
    "lr": 1e-3,
    "batch_size": 64,
    "max_epochs": 100,
    "weight_decay": 1e-6,
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


def train_eyepacs(contrastive=False, label_transform=False):
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

    model = EyePACS_Model(hparams).to(DEVICE)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=hparams["max_epochs"],
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        sync_batchnorm=True,
        strategy="ddp_find_unused_parameters_false",
        accelerator="gpu",
        devices="auto",
        precision=16,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


def train_vindr(contrastive=False, label_transform=False):
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

    model = VinDR_Model(hparams).to(DEVICE)
    trainer = pl.Trainer(
        max_epochs=hparams["max_epochs"],
        devices="auto",
        accelerator="gpu",
        precision=16,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":

    if hparams["dataset"] == "EyePACS":
        print("Training for EyePACS...")
        train_eyepacs(contrastive=hparams["contrastive"], label_transform=True)
    elif hparams["dataset"] == "VinDR":
        print("Training for VinDR...")
        train_vindr(contrastive=False, label_transform=True)
