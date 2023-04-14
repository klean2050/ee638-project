import torch, torch.nn as nn, torch.optim as optim
from pytorch_lightning import LightningModule
from torchvision import models
from simclr.modules import NT_Xent


class EyePACS_Model(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # backbone encoder
        densenet = models.densenet121(pretrained=True)
        densenet.fc = nn.Identity()
        self.encoder = densenet
        for param in self.encoder.parameters():
            param.requires_grad = False

        # projection heads
        output_dim = 128 if self.hparams.contrastive else 5
        self.projector = nn.Sequential(
            nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, output_dim)
        )

        # loss functions
        self.ssl_loss = NT_Xent(2 * self.hparams.batch_size, 0.1, world_size=1)
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        embedding = self.encoder(x)
        return self.projector(embedding)

    def training_step(self, batch, _):

        if self.hparams.contrastive:
            x_i, x_j, y = batch
            z_i = self.forward(x_i)
            z_j = self.forward(x_j)
            loss = self.ssl_loss(z_i, z_j)
        else:
            x, y = batch
            preds = self.forward(x)
            # calculate class weights
            weights = torch.bincount(y).float()
            weights = [weights.sum() / w for w in weights]
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            loss = loss_fn(preds, y)

        self.log("Train/loss", loss, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, _):

        if self.hparams.contrastive:
            x_i, x_j, y = batch
            z_i = self.forward(x_i)
            z_j = self.forward(x_j)
            loss = self.ssl_loss(z_i, z_j)
        else:
            x, y = batch
            preds = self.forward(x)
            loss = self.cls_loss(preds, y)

        self.log("Valid/loss", loss, sync_dist=True, batch_size=self.hparams.batch_size)

        # compute accuracy
        if not self.hparams.contrastive:
            preds = preds.argmax(dim=1)
            acc = (preds == y).float().mean()
            self.log(
                "Valid/acc", acc, sync_dist=True, batch_size=self.hparams.batch_size
            )

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class VinDR_Model(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # backbone encoder
        densenet = models.densenet121(pretrained=True)
        densenet.fc = nn.Identity()
        self.encoder = densenet
        for param in self.encoder.parameters():
            param.requires_grad = False

        # projection heads
        output_dim = 128 if self.hparams.contrastive else 5
        self.projector = nn.Sequential(
            nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, output_dim)
        )

        # loss functions
        self.ssl_loss = NT_Xent(2 * self.hparams.batch_size, 0.1, world_size=1)
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        embedding = self.encoder(x)
        return self.projector(embedding)

    def training_step(self, batch, _):

        if self.hparams.contrastive:
            x_i, x_j, y = batch
            z_i = self.forward(x_i)
            z_j = self.forward(x_j)
            return self.ssl_loss(z_i, z_j)
        else:
            x, y = batch
            preds = self.forward(x)
            return self.cls_loss(preds, y)

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


if __name__ == "__main__":
    hparams = {
        "dataset": "EyePACS",
        "lr": 1e-3,
        "batch_size": 64,
        "max_epochs": 100,
        "weight_decay": 1e-6,
    }
    model = EyePACS_Model(hparams)
    print(model)
