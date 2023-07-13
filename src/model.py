import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class SentimentNet(nn.Module):
    def __init__(
        self,
        pretrained,
        backbone_dropout=0.0,
        backbone_layer_norm=1e-7,
        cls_dropout=0.0,
        cls_hidden=512,
    ):
        super().__init__()
        self.pretrained_config = AutoConfig.from_pretrained(pretrained)
        self.pretrained_config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": backbone_dropout,
                "layer_norm_eps": backbone_layer_norm,
            }
        )
        self.backbone = AutoModel.from_pretrained(
            pretrained, config=self.pretrained_config
        )
        self.feature_dim = self.backbone.config.hidden_size
        self.backbone.pooler = None

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, cls_hidden),
            nn.Dropout(p=cls_dropout) if cls_dropout > 0.0 else nn.Identity(),
            nn.ReLU(),
            nn.Linear(cls_hidden, 1),
        )

    def get_tweet_embeddings(self, input_ids, attention_mask):
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).hidden_states

    def forward(self, input_ids, attention_mask):
        layers = self.get_tweet_embeddings(input_ids, attention_mask)
        last_layer = layers[-1]
        cls_token = last_layer[:, 0, :]
        logit = self.classifier(cls_token)
        return logit
