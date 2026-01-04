import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embed_dim=16,
        hidden_layers=(64, 32),
        dropout=0.5
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        layers = []
        prev_dim = 2 * embed_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        x = torch.cat([user_emb, item_emb], dim=-1)
        x = self.mlp(x)

        logit = self.output(x).squeeze(-1)
        prob = torch.sigmoid(logit)
        return prob
