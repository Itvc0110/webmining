import torch
from torch import nn
import torch.nn.functional as F

class ExponentialCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_cross_layers=3,
                 layer_norm=True,
                 batch_norm=True,
                 net_dropout=0.1):
        super(ExponentialCrossNetwork, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        self.gates = nn.ModuleList()
        
        # compute split size
        self.input_dim = input_dim
        self.half = input_dim // 2

        for i in range(num_cross_layers):
            w_layer = nn.Linear(input_dim, self.half, bias=False).to(self.device)
            self.w.append(w_layer)
            self.b.append(nn.Parameter(torch.zeros((input_dim,), device=self.device)))

            gate_layer = nn.Linear(self.half, self.half).to(self.device)
            self.gates.append(gate_layer)

            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(self.half).to(self.device))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(self.half).to(self.device))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout).to(self.device))
            nn.init.uniform_(self.b[i].data)
            
        self.masker = nn.ReLU().to(self.device)

        self.dfc = nn.Linear(input_dim, 1).to(self.device)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = x.to(self.device)

        for i in range(self.num_cross_layers):
            H = self.w[i](x)

            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)

            gate = torch.sigmoid(self.gates[i](H))
            H = H * gate 

            # concatenate and pad if necessary
            H = torch.cat([H, H * mask], dim=-1)

            if H.shape[-1] != self.input_dim:
                pad_size = self.input_dim - H.shape[-1]
                pad = H.new_zeros(H.size(0), pad_size)
                H = torch.cat([H, pad], dim=-1)
            x = x * (H + self.b[i]) + x
            x = torch.clamp(x, min=-10, max=10)
            x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-12)
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        logit = self.dfc(x)
        return logit.squeeze()  


class LinearCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_cross_layers=3,
                 layer_norm=True,
                 batch_norm=True,
                 net_dropout=0.1):
        super(LinearCrossNetwork, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        self.gates = nn.ModuleList()

        # compute split size
        self.input_dim = input_dim
        self.half = input_dim // 2
        
        for i in range(num_cross_layers):
            w_layer = nn.Linear(input_dim, self.half, bias=False).to(self.device)
            self.w.append(w_layer)
            self.b.append(nn.Parameter(torch.zeros((input_dim,), device=self.device)))

            gate_layer = nn.Linear(self.half, self.half).to(self.device)
            self.gates.append(gate_layer)
            
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(self.half).to(self.device))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(self.half).to(self.device))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout).to(self.device))
            nn.init.uniform_(self.b[i].data)
            
        self.masker = nn.ReLU().to(self.device)

        self.sfc = nn.Linear(input_dim, 1).to(self.device)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = x.to(self.device)
        x0 = x

        for i in range(self.num_cross_layers):
            H = self.w[i](x)
    
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)

            gate = torch.sigmoid(self.gates[i](H))
            H = H * gate        

            H = torch.cat([H, H * mask], dim=-1)
            if H.shape[-1] != self.input_dim:
                pad_size = self.input_dim - H.shape[-1]
                pad = H.new_zeros(H.size(0), pad_size)
                H = torch.cat([H, pad], dim=-1)

            x = x0 * (H + self.b[i]) + x
            x = torch.clamp(x, min=-10, max=10)
            x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-12)
            if len(self.dropout) > i:
                x = self.dropout[i](x)
                
        logit = self.sfc(x)
        return logit.squeeze()  # Changed to squeeze


class DCNv3(nn.Module):
    def __init__(self, field_dims, dense_dim=18, embed_dim=16, num_deep_cross_layers=2, num_shallow_cross_layers=4, deep_net_dropout=0.1, shallow_net_dropout=0.1, layer_norm=True, batch_norm=True):
        super(DCNv3, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Embeddings for sparse fields
        self.embeddings = nn.ModuleList([nn.Embedding(fd, embed_dim) for fd in field_dims])
        
        # Dense projection
        self.dense_proj = nn.Linear(dense_dim, embed_dim)
        
        self.input_dim = len(field_dims) * embed_dim + embed_dim  
        
        self.ECN = ExponentialCrossNetwork(self.input_dim, num_deep_cross_layers, net_dropout=deep_net_dropout, layer_norm=layer_norm, batch_norm=batch_norm).to(self.device)
        self.LCN = LinearCrossNetwork(self.input_dim, num_shallow_cross_layers, net_dropout=shallow_net_dropout, layer_norm=layer_norm, batch_norm=batch_norm).to(self.device)
        
        self.apply(self._init_weights)
        self.output_activation = torch.sigmoid

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, sparse, dense):
        sparse, dense = sparse.to(self.device), dense.to(self.device)
        
        # Sparse embeds
        sparse_embeds = [self.embeddings[i](sparse[:, i]) for i in range(sparse.size(1))]
        sparse_concat = torch.cat(sparse_embeds, dim=1)
        
        # Dense proj
        dense_emb = self.dense_proj(dense)
        
        # Combined
        feature_emb = torch.cat([sparse_concat, dense_emb], dim=1)
        
        dlogit = self.ECN(feature_emb)
        slogit = self.LCN(feature_emb)
        logit = (dlogit + slogit) * 0.5
        
        y_pred = self.output_activation(logit)
        y_d = self.output_activation(dlogit)
        y_s = self.output_activation(slogit)
        return {"y_pred": y_pred, "y_d": y_d, "y_s": y_s}
