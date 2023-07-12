import os
import math
import wandb
import datetime
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import AutoTokenizer
from dataloader import TwitterDataModule
from model import SentimentNet
from utils import cos_anneal


class LightningModel(pl.LightningModule):
    def __init__(self, **config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.train_config = config["train"]
        self.model_config = config["model"]
        self.model = SentimentNet(**self.model_config)
    
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean") # Numerically stable than BCELoss   

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x["input_ids"], x["attention_mask"])
    
    def training_step(self, batch, batch_idx):
        x = batch
        y = batch["label"]
        logit = self(x)
        loss = self.loss_fn(logit, y)
        pred = (F.sigmoid(logit) > 0.5).long().detach()
        accuracy = (pred == y).float().mean()
        self.log("train/loss", loss.item())
        self.log("train/accuracy", accuracy.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        y = batch["label"]
        logit = self(x)
        loss = self.loss_fn(logit, y)
        pred = (F.sigmoid(logit) > 0.5).long().detach()
        accuracy = (pred == y).float().mean()
        self.log("val/loss", loss.item())
        self.log("val/accuracy", accuracy.item())

    def configure_optimizers(self):
        # Layerwise learning rate decay for the Transformers backbone
        no_weight_decay = ["bias", "LayerNorm.weight"]
        params = list(self.model.named_parameters())
        is_backbone = lambda n: "backbone" in n and "pooler" not in n
        backbone_param_names = [n for n, p in params if is_backbone(n)]
        backbone_param_names.reverse()
        backbone_lr = self.train_config["lr"]
        lr_decay = self.train_config["layerwise_lr_decay"]
        param_groups = []
        for idx, name in enumerate(backbone_param_names):
            # print(f'{idx}: lr = {backbone_lr:.6f}, {name}')
            param_groups.append(
                {
                    'params': [p for n, p in params if n == name and n not in no_weight_decay and p.requires_grad],
                    'lr': backbone_lr,
                    "weight_decay": self.train_config["weight_decay"],
                    "betas": (0.9, 0.999)
                }
            )
            param_groups.append(
                {
                    'params': [p for n, p in params if n == name and n in no_weight_decay and p.requires_grad],
                    'lr': backbone_lr,
                    "weight_decay": 0,
                    "betas": (0.9, 0.999)
                }
            )
            backbone_lr *= lr_decay

        # Classifier head
        param_groups.append(
            {
                "params": [p for n, p in params if not is_backbone(n)]
            }
        )

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.train_config["lr"],
            betas=(0.9, 0.999),
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
        # Linear warm up then cosine decay
        for i, g in enumerate(self.optimizer.param_groups):
            if current_step < self.warmup_steps:
                g["lr"] = current_step / self.warmup_steps * self.default_lrs[i]
            else:
                g["lr"] = cos_anneal(
                    self.warmup_steps, 
                    self.total_steps,
                    self.default_lrs[i],
                    self.train_config["min_lr"],
                    self.global_step
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
        precision=32, # Use fp32 for now
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