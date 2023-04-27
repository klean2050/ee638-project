import os, torch, pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms import *
from torchvision import models

from datasets import EyePACS, VinDR
from models import SSLModel, SupervisedModel, CombinedModel

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hparams = {
    "dataset": "VinDR",  # must be either "EyePACS" or "VinDR"
    "contrastive": True,
    "lr": 1e-3,
    "batch_size": 64,
    "max_epochs": 100,
    "weight_decay": 1e-6,
    "distort": True,
    "encoder": models.densenet121(pretrained=True),
    "combine": False,  # if True, "contrastive" must be True
}
root = (
    "/data/avramidi/tiny_vindr/"
    if hparams["dataset"] == "VinDR"
    else "/data/avramidi/large_fundus/"
)
data_transform = Compose(
    [
        RandomResizedCrop((224, 224)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ColorJitter(brightness=0.2, saturation=0.2, hue=0.2),
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
        drop_last=True,
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

    if not hparams["contrastive"]:
        model = SupervisedModel(hparams).to(device)
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
            trainer.test(model, dataloaders=test_loader)
    elif not hparams["combine"]:
        model = SSLModel(hparams).to(device)
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=hparams["max_epochs"] // 10,
            check_val_every_n_epoch=1,
            log_every_n_steps=5,
            sync_batchnorm=True,
            strategy="ddp_find_unused_parameters_false",
            accelerator="gpu",
            devices="auto",
            precision=16,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

        pretrained_model = SSLModel.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        ).to(device)
        hparams["encoder"] = pretrained_model.encoder
        model = SupervisedModel(hparams).to(device)

        # change datasets to supervised
        train_dataset.contrastive = False
        valid_dataset.contrastive = False
        if test_dataset is not None:
            test_dataset.contrastive = False

        trainer = pl.Trainer(
            logger=logger,
            max_epochs=hparams["max_epochs"] // 2,
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
            trainer.test(model, dataloaders=test_loader)
    else:
        model = CombinedModel(hparams).to(device)
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
            trainer.test(model, dataloaders=test_loader)
