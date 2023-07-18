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
        monte_carlo_dropout=True,
        dropout_rate=0.5,
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
        # self.backbone.pooler = None # Disable pooled_output

        if monte_carlo_dropout is True:
            self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(5)])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate)])

        # self.classifier = nn.Linear(self.feature_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outs["pooler_output"]
        embedding = pooled_output
        # layers = self.get_tweet_embeddings(input_ids, attention_mask)
        # last_layer = layers[-1]
        # cls_token = last_layer[:, 0, :]
        # mean_pooled = last_layer.mean(dim=1)
        # embedding = mean_pooled
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logit = self.classifier(dropout(embedding))
            else:
                logit += self.classifier(dropout(embedding))

        logit /= len(self.dropouts)
        return logit
