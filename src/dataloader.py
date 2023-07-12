import os
import re
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

import pytorch_lightning as pl


class TwitterDataset(Dataset):
    def __init__(self, root, tokenizer, max_length=140, mode="train"):
        self.root = root
        self.tokenizer = tokenizer
        # self.tokenizer.add_tokens("<user>")
        # self.tokenizer.add_tokens("<url>")
        self.max_length = max_length
        self.mode = mode
        self.df = pd.read_csv(root)
        self.tweets = [re.sub(r"\s{2,}", " ", tweet) for tweet in self.df["tweet"]]
        self.labels = self.df["label"] if mode != "test" else None
        self.indices = np.arange(len(self.df))

        rng = np.random.default_rng(42)
        rng.shuffle(self.indices)
        if self.mode == "train":
            self.indices = self.indices[:int(0.9 * len(self.indices))]
            
        elif self.mode == "val":
            self.indices = self.indices[int(0.9 * len(self.indices)):]
        
        # Sort by the tweet's length to make batching faster
        tweet_lengths = np.array([len(self.tweets[i]) for i in self.indices])
        sorted_indices = np.argsort(tweet_lengths)
        self.indices = self.indices[sorted_indices]

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        tweet = self.tweets[self.indices[idx]]
        label = self.labels[self.indices[idx]] if self.mode != "test" else None
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
                "token_type_ids": enc["token_type_ids"].long(),
                "attention_mask": enc["attention_mask"].long(),
            }
        else:
            return {
                "tweet": tweet,
                "input_ids": enc["input_ids"].long(),
                "token_type_ids": enc["token_type_ids"].long(),
                "attention_mask": enc["attention_mask"].long(),
                "label": torch.tensor(label, dtype=torch.float32),
            }

class TwitterDataModule(pl.LightningDataModule):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.pretrained = config["pretrained"]
        self.data_dir = config["data_dir"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.num_gpus = len(config["gpu_ids"])

    def setup(self, stage=None):
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained)
        self.train_data = TwitterDataset(
            root=self.data_dir,
            tokenizer=tokenizer,
            mode="train",
        )
        self.val_data = TwitterDataset(
            root=self.data_dir,
            tokenizer=tokenizer,
            mode="val",
        )
    
    def collate_fn(self, batch):
        input_ids = [item["input_ids"].squeeze() for item in batch]
        token_type_ids = [item["token_type_ids"].squeeze() for item in batch]
        attention_mask = [item["attention_mask"].squeeze() for item in batch]
        labels = [item["label"] for item in batch] if "label" in batch[0] else None

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        batch = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(labels).unsqueeze(1) if labels else None,
        }
        return batch
    
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size // self.num_gpus,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size // self.num_gpus,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
