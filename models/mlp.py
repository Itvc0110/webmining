import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=16, hidden_layers=[64, 32], dropout=0.5):
        super(MLP, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.user_embedding = nn.Embedding(num_users, embed_dim).to(self.device)
        self.item_embedding = nn.Embedding(num_items, embed_dim).to(self.device)
        
        input_dim = 2 * embed_dim
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, hidden_dim).to(self.device))
            self.layers.append(nn.BatchNorm1d(hidden_dim).to(self.device))
            self.layers.append(nn.ReLU().to(self.device))
            self.layers.append(nn.Dropout(dropout).to(self.device))
            prev_dim = hidden_dim
        self.output = nn.Linear(prev_dim, 1).to(self.device)

    def forward(self, user_ids, item_ids):
        user_ids = user_ids.to(self.device)
        item_ids = item_ids.to(self.device)
        
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        for layer in self.layers:
            x = layer(x)
        logit = self.output(x).squeeze()
        return torch.sigmoid(logit)