import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFM(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=16, hidden_layers=[32, 32], dropout=0.5):
        super(DeepFM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.user_embedding = nn.Embedding(num_users, embed_dim).to(self.device)
        self.item_embedding = nn.Embedding(num_items, embed_dim).to(self.device)
        
        self.user_linear = nn.Embedding(num_users, 1).to(self.device)
        self.item_linear = nn.Embedding(num_items, 1).to(self.device)
        
        input_dim = 2 * embed_dim
        self.dnn_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            self.dnn_layers.append(nn.Linear(prev_dim, hidden_dim).to(self.device))
            self.dnn_layers.append(nn.BatchNorm1d(hidden_dim).to(self.device))
            self.dnn_layers.append(nn.ReLU().to(self.device))
            self.dnn_layers.append(nn.Dropout(dropout).to(self.device))
            prev_dim = hidden_dim
        self.dnn_output = nn.Linear(prev_dim, 1).to(self.device)

        self.bias = nn.Parameter(torch.zeros(1).to(self.device))

    def forward(self, user_ids, item_ids):
        user_ids = user_ids.to(self.device)
        item_ids = item_ids.to(self.device)
        
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        emb_concat = torch.cat([user_emb, item_emb], dim=-1)
        
        user_linear = self.user_linear(user_ids)
        item_linear = self.item_linear(item_ids)
        linear = user_linear + item_linear
        
        emb_list = [user_emb, item_emb]
        sum_emb = torch.sum(torch.stack(emb_list), dim=0)
        sum_emb_square = torch.square(sum_emb)
        square_emb_sum = torch.sum(torch.stack([torch.square(e) for e in emb_list]), dim=0)
        second_order = 0.5 * (sum_emb_square - square_emb_sum)
        
        dnn_out = emb_concat
        for layer in self.dnn_layers:
            dnn_out = layer(dnn_out)
        dnn_out = self.dnn_output(dnn_out)
        
        logit = linear.squeeze() + torch.sum(second_order, dim=1) + dnn_out.squeeze() + self.bias
        return torch.sigmoid(logit)