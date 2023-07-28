import datetime
import math
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from dataloader import TwitterDataModule
from model import SentimentNet
from utils import cos_anneal


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class LightningModel(pl.LightningModule):
    def __init__(self, **config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.train_config = config["train"]
        self.model_config = config["model"]
        self.model = SentimentNet(**self.model_config)

        # self.loss_fn = nn.BCEWithLogitsLoss(
        #     reduction="mean"
        # )  # Numerically stable than BCELoss
        self.loss_fn = SmoothBCEwLogits(smoothing=0.1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x["input_ids"], x["attention_mask"])

    def training_step(self, batch, batch_idx):
        x = batch
        y = batch["label"]
        logit = self(x)
        loss = self.loss_fn(logit, y)
        pred = (F.sigmoid(logit) > 0.5).float().detach()
        accuracy = (pred == y).float().mean()
        self.log("train/loss", loss.item())
        self.log("train/accuracy", accuracy.item())
        return loss

    def validation_step(self, batch, batch_idx):
        if hasattr(self.model, "dropouts"):
            self.model.dropouts.train()  # Monte-Carlo dropout

        x = batch
        y = batch["label"]
        logit = self(x)
        loss = self.loss_fn(logit, y)
        pred = (F.sigmoid(logit) > 0.5).float().detach()
        accuracy = (pred == y).float().mean()
        self.log("val/loss", loss.item(), sync_dist=True)
        self.log("val/accuracy", accuracy.item(), sync_dist=True)

    def predict_step(self, batch, batch_idx):
        if hasattr(self.model, "dropouts"):
            self.model.dropouts.train()  # Monte-Carlo dropout

        logit = self(batch)
        if batch["label"] is None:
            return logit
        else:
            return logit, batch["label"]

    def configure_optimizers(self):
        # Layerwise learning rate decay for the Transformers backbone
        no_weight_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        params = list(self.model.named_parameters())
        is_backbone = lambda n: "backbone" in n and "pooler" not in n
        backbone_param_names = [n for n, p in params if is_backbone(n)]
        backbone_param_names.reverse()
        backbone_lr = self.train_config["backbone_lr"]
        lr_decay = self.train_config["layerwise_lr_decay"]
        param_groups = []
        for idx, name in enumerate(backbone_param_names):
            # print(f'{idx}: lr = {backbone_lr:.6f}, {name}')
            param_groups.append(
                {
                    "params": [
                        p
                        for n, p in params
                        if n == name and n not in no_weight_decay and p.requires_grad
                    ],
                    "lr": backbone_lr,
                    "weight_decay": self.train_config["weight_decay"],
                    "betas": (0.9, 0.999),
                }
            )
            param_groups.append(
                {
                    "params": [
                        p
                        for n, p in params
                        if n == name and n in no_weight_decay and p.requires_grad
                    ],
                    "lr": backbone_lr,
                    "weight_decay": 0,
                    "betas": (0.9, 0.999),
                }
            )
            backbone_lr *= lr_decay

        param_groups.append(
            {
                "params": [p for n, p in params if not is_backbone(n)],
                "lr": self.train_config["classifier_lr"],
                "betas": (0.9, 0.999),
            }
        )

        self.optimizer = torch.optim.AdamW(
            param_groups,
        )

        self.default_lrs = []
        for g in self.optimizer.param_groups:
            self.default_lrs.append(g["lr"])

        return self.optimizer

    def on_train_batch_start(self, batch, batch_idx):
        num_batches = len(self.trainer.train_dataloader)
        self.total_steps = num_batches * self.train_config["max_epochs"]
        self.warmup_steps = int(self.total_steps * self.train_config["lr_warmup_pct"])
        current_step = self.global_step

        lr = 0
        for i, g in enumerate(self.optimizer.param_groups):
            if current_step < self.warmup_steps:
                g["lr"] = current_step / self.warmup_steps * self.default_lrs[i]
            else:
                # g["lr"] = cos_anneal(
                #     self.warmup_steps,
                #     self.total_steps,
                #     self.default_lrs[i],
                #     self.train_config["min_lr"],
                #     self.global_step,
                # )
                g["lr"] = (
                    (self.total_steps - current_step)
                    / (self.total_steps - self.warmup_steps)
                    * self.default_lrs[i]
                )
            lr = max(lr, g["lr"])

        self.log("train/lr", lr)


def main(**args):
    assert args["run_id"] != "", "Please specify RUN_ID!"
    config = OmegaConf.load(args["config_path"])

    run_id = args["run_id"]
    debug = args["debug"]
    config_name = os.path.splitext(args["config_path"])[0].split("/")[1]
    log_dir = f"./outputs/{config_name}/{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    logger = WandbLogger(
        project="ethz-cil-project",
        entity="kvu207",
        name=run_id,
        save_dir=log_dir,
        offline=debug,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=log_dir,
        filename=f"{run_id}",
        save_top_k=1,
        mode="min",
    )

    data = TwitterDataModule(**config["train"])
    model = LightningModel(**config)
    print(model)

    callback_list = [checkpoint_callback] if not debug else []

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config["train"]["gpu_ids"],
        deterministic=False,
        precision=32,  # Use fp32 for now
        strategy="auto",
        detect_anomaly=debug,
        log_every_n_steps=1 if debug else 50,
        fast_dev_run=5 if debug else False,
        default_root_dir=log_dir,
        max_epochs=config["train"]["max_epochs"],
        val_check_interval=0.25,
        enable_progress_bar=debug,
        logger=False if debug else logger,
        enable_checkpointing=not debug,
        callbacks=callback_list,
    )
    trainer.fit(model, data)
    wandb.finish()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
