import os
import re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TwitterDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=164, mode="train"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.data_path = f"data/{mode}/{dataset}.csv"
        self.df = pd.read_csv(self.data_path)
        self.tweets = self.df["tweet"]
        self.labels = self.df["label"] if mode != "test" else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        label = self.labels[idx] if self.mode != "test" else None
        enc = self.tokenizer.encode_plus(
            tweet,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if self.mode == "test":
            return {
                "tweet": tweet,
                "input_ids": enc["input_ids"].long(),
                "attention_mask": enc["attention_mask"].long(),
            }
        else:
            return {
                "tweet": tweet,
                "input_ids": enc["input_ids"].long(),
                "attention_mask": enc["attention_mask"].long(),
                "label": torch.tensor(label, dtype=torch.float32),
            }


def longest_length_padding(batch):
    input_ids = [item["input_ids"].squeeze() for item in batch]
    attention_mask = [item["attention_mask"].squeeze() for item in batch]
    labels = [item["label"] for item in batch] if "label" in batch[0] else None

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": torch.tensor(labels).unsqueeze(1) if labels else None,
    }
    return batch


class TwitterDataModule(pl.LightningDataModule):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.pretrained = config["pretrained"]
        self.dataset = config["dataset"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.num_gpus = len(config["gpu_ids"])

    def setup(self, stage=None):
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained)
        self.train_data = TwitterDataset(
            self.dataset,
            tokenizer=tokenizer,
            mode="train",
        )
        self.val_data = TwitterDataset(
            self.dataset,
            tokenizer=tokenizer,
            mode="val",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size // self.num_gpus,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=longest_length_padding,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size // self.num_gpus,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=longest_length_padding,
        )
