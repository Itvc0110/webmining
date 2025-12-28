import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from config import get_args
from movielens import MovieLens1MDataset, MovieLens1MDatasetWithMetadata
from models.dcnv3 import DCNv3
from models.deepfm import DeepFM
from models.mlp import MLP
from loss import TriBCE_Loss, Weighted_TriBCE_Loss, BCE_Loss, Weighted_BCE_Loss
import torch.nn as nn

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    pin_memory = True if device.type == 'cuda' else False

    # Dataset
    if args.model_type == 'dcnv3':
        dataset = MovieLens1MDatasetWithMetadata(args.dataset_path, args.users_path, args.movies_path)
    else:
        dataset = MovieLens1MDataset(args.dataset_path)
    
    # Load dataset
    num_users = len(dataset.user_ids())
    num_items = len(dataset.movie_ids())
    
    # Split indices 
    indices = np.arange(len(dataset))
    train_indices, temp_indices = train_test_split(indices, test_size=1 - args.train_ratio, random_state=args.seed)
    val_indices, test_indices = train_test_split(temp_indices, test_size=(1 - args.train_ratio - args.val_ratio) / (1 - args.train_ratio), random_state=args.seed)
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
 
    
    # Save test indices 
    np.save('test_indices.npy', test_indices)
    
    # Initialize model
    if args.model_type == 'dcnv3':
        model = DCNv3(
            field_dims=dataset.field_dims,   
            dense_dim=dataset.dense_dim,     
            embed_dim=args.embed_dim,
            num_deep_cross_layers=args.num_deep_cross_layers,
            num_shallow_cross_layers=args.num_shallow_cross_layers,
            deep_net_dropout=args.deep_net_dropout,
            shallow_net_dropout=args.shallow_net_dropout,
            layer_norm=args.layer_norm,
            batch_norm=args.batch_norm
        ).to(device)
    elif args.model_type == 'deepfm':
        hidden_layers = [int(x) for x in args.deepfm_hidden_layers.split(',')]
        model = DeepFM(
            num_users=num_users, 
            num_items=num_items, 
            embed_dim=args.embed_dim, 
            hidden_layers=hidden_layers,
            dropout=args.dropout
        ).to(device)
    elif args.model_type == 'mlp':
        hidden_layers = [int(x) for x in args.mlp_hidden_layers.split(',')]
        model = MLP(
            num_users=num_users, 
            num_items=num_items, 
            embed_dim=args.embed_dim, 
            hidden_layers=hidden_layers,
            dropout=args.dropout
        ).to(device)
    
    # Loss
    if args.loss_type == 'tribce':
        criterion = TriBCE_Loss()
    elif args.loss_type == 'weighted_tribce':
        criterion = Weighted_TriBCE_Loss()
    elif args.loss_type == 'bce':
        criterion = BCE_Loss()
    elif args.loss_type == 'weighted_bce':
        criterion = Weighted_BCE_Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            items, targets = batch
            user_ids, item_ids = items[:, 0], items[:, 1]
            targets = targets.to(device).float()
            
            optimizer.zero_grad()
            if args.model_type == 'dcnv3':
                sparse, dense, targets = batch
                sparse, dense, targets = sparse.to(device), dense.to(device), targets.to(device).float()
                outputs = model(sparse, dense)
                loss = criterion(outputs['y_pred'], targets, outputs['y_d'], outputs['y_s'])
            else:
                items, targets = batch
                user_ids, item_ids = items[:, 0].to(device), items[:, 1].to(device)
                targets = targets.to(device).float()
                outputs = model(user_ids, item_ids)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                items, targets = batch
                user_ids, item_ids = items[:, 0], items[:, 1]
                targets = targets.to(device).float()
                
                if args.model_type == 'dcnv3':
                    outputs = model(user_ids, item_ids)
                    loss = criterion(outputs['y_pred'], targets, outputs['y_d'], outputs['y_s'])
                else:
                    outputs = model(user_ids, item_ids)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print('Early stopping triggered')
                break

if __name__ == '__main__':

    main()



