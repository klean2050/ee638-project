import torch.nn as nn, torch.optim as optim
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score
from simclr.modules import NT_Xent


class SSLModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        # backbone encoder
        self.encoder = self.hparams.encoder
        self.encoder.fc = nn.Identity()
        # projection heads
        self.projector = nn.Sequential(
            nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 128)
        )
        # loss functions
        self.ssl_loss = NT_Xent(self.hparams.batch_size, 0.1, world_size=1)

    def forward(self, x):
        embedding = self.encoder(x)
        return self.projector(embedding)

    def training_step(self, batch, _):
        x_i, x_j, _ = batch
        z_i = self.forward(x_i)
        z_j = self.forward(x_j)
        loss = self.ssl_loss(z_i, z_j)
        self.log("Train/loss", loss, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, _):
        x_i, x_j, _ = batch
        z_i = self.forward(x_i)
        z_j = self.forward(x_j)
        loss = self.ssl_loss(z_i, z_j)
        self.log("Valid/loss", loss, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class SupervisedModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.nc = 8 if self.hparams.dataset == "VinDR" else 5
        # backbone encoder
        self.encoder = self.hparams.encoder
        # projection heads
        self.classifier = nn.Sequential(
            nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, self.nc)
        )
        # loss functions
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        embedding = self.encoder(x)
        return self.classifier(embedding)

    def training_step(self, batch, _):
        x, y = batch
        preds = self.forward(x)
        loss = self.cls_loss(preds, y)
        self.log("Train/loss", loss, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        preds = self.forward(x)
        loss = self.cls_loss(preds, y)
        self.log("Valid/loss", loss, sync_dist=True, batch_size=self.hparams.batch_size)

        # compute accuracy
        preds = preds.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("Valid/acc", acc, sync_dist=True, batch_size=self.hparams.batch_size)
        f1 = f1_score(y.cpu(), preds.cpu(), average="macro")
        self.log("Valid/f1", f1, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def test_step(self, batch, _):
        x, y = batch
        preds = self.forward(x).argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("Test/acc", acc, sync_dist=True, batch_size=self.hparams.batch_size)
        f1 = f1_score(y.cpu(), preds.cpu(), average="macro")
        self.log("Test/f1", f1, sync_dist=True, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class CombinedModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.nc = 8 if self.hparams.dataset == "VinDR" else 5
        # backbone encoder
        self.encoder = self.hparams.encoder
        # projection heads
        self.projector = nn.Sequential(
            nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, self.nc)
        )
        # loss functions
        self.cls_loss = nn.CrossEntropyLoss()
        self.ssl_loss = NT_Xent(self.hparams.batch_size, 0.1, world_size=1)

    def forward(self, x):
        embedding = self.encoder(x)
        return self.classifier(embedding), self.projector(embedding)

    def training_step(self, batch, _):
        x_i, x_j, y = batch
        preds_i, embeds_i = self.forward(x_i)
        preds_j, embeds_j = self.forward(x_j)
        loss = self.cls_loss(preds_i, y)
        loss += self.cls_loss(preds_j, y)
        loss += 0.5 * self.ssl_loss(embeds_i, embeds_j)
        self.log("Train/loss", loss, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, _):
        x_i, x_j, y = batch
        preds_i, embeds_i = self.forward(x_i)
        preds_j, embeds_j = self.forward(x_j)
        loss = self.cls_loss(preds_i, y)
        loss += self.cls_loss(preds_j, y)
        loss += 0.5 * self.ssl_loss(embeds_i, embeds_j)
        self.log("Valid/loss", loss, sync_dist=True, batch_size=self.hparams.batch_size)

        # compute accuracy
        preds = preds_i.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("Valid/acc", acc, sync_dist=True, batch_size=self.hparams.batch_size)
        f1 = f1_score(y.cpu(), preds.cpu(), average="macro")
        self.log("Valid/f1", f1, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def test_step(self, batch, _):
        x_i, _, y = batch
        preds = self.forward(x_i).argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("Test/acc", acc, sync_dist=True, batch_size=self.hparams.batch_size)
        f1 = f1_score(y.cpu(), preds.cpu(), average="macro")
        self.log("Test/f1", f1, sync_dist=True, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


if __name__ == "__main__":
    import torch
    from torchvision import models

    hparams = {
        "dataset": "EyePACS",
        "lr": 1e-3,
        "batch_size": 64,
        "weight_decay": 1e-6,
        "encoder": models.densenet121(pretrained=True),
    }
    sample = torch.randn(2, 3, 224, 224)
    print(CombinedModel(hparams)(sample))
