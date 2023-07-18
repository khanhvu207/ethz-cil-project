import math
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from dataloader import TwitterDataset, longest_length_padding
from train import LightningModel


def main(**args):
    run_id = args["run_id"]
    ckpt_path = args["ckpt_path"]
    config = OmegaConf.load(args["config_path"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained"])
    val_dataset = TwitterDataset(
        config["train"]["data_dir"],
        tokenizer,
        mode="val",
    )
    test_dataset = TwitterDataset(
        config["predict"]["test_dir"],
        tokenizer,
        mode="test",
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
        collate_fn=longest_length_padding,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
        collate_fn=longest_length_padding,
    )
    model = LightningModel.load_from_checkpoint(ckpt_path)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        enable_progress_bar=True,
        # limit_predict_batches=10,
        logger=False,
    )
    val_outs = trainer.predict(model, val_dataloader)
    test_outs = trainer.predict(model, test_dataloader)

    val_preds = []
    val_labels = []
    for logits, labels in val_outs:
        val_preds.append(F.sigmoid(logits))
        val_labels.append(labels)

    test_preds = []
    for logits in test_outs:
        test_preds.append(F.sigmoid(logits))

    val_preds = torch.vstack(val_preds)
    val_labels = torch.vstack(val_labels)
    test_preds = torch.vstack(test_preds)

    # Postprocessing
    best_t = 0.5
    best_acc = 0
    print("Probing for an optimal decision boundary on the validation set...")
    for t in np.arange(0.1, 1.0, 0.05):
        y_hat = (val_preds > t).float()
        acc = (y_hat == val_labels).float().mean()
        print(f"Threshold {t:.3f}: accuracy={acc:.5f}")
        if best_acc < acc:
            best_acc = acc
            best_t = t

    test_calibrated_preds = (test_preds > best_t).int() * 2 - 1

    print("Test prediction statistics")
    print("Number of negative predictions:", (test_calibrated_preds == -1).sum())
    print("Number of positive predictions:", (test_calibrated_preds == 1).sum())

    val_pred_df = pd.DataFrame()
    test_pred_df = pd.DataFrame()
    test_raw_pred = pd.DataFrame()
    val_pred_df["pred_probability"] = val_preds.squeeze().tolist()
    val_pred_df["label"] = val_labels.squeeze().tolist()
    test_raw_pred["pred_probability"] = test_preds.squeeze().tolist()
    test_pred_df["Id"] = np.arange(1, test_calibrated_preds.shape[0] + 1)
    test_pred_df["Prediction"] = test_calibrated_preds.squeeze().tolist()

    save_dir = os.path.dirname(ckpt_path)
    val_pred_df.to_csv(os.path.join(save_dir, "val_pred.csv"), index=False)
    test_pred_df.to_csv(os.path.join(save_dir, "submission.csv"), index=False)
    test_raw_pred.to_csv(os.path.join(save_dir, "test_pred.csv"), index=False)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
