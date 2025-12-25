# evaluate.py (updated with pin_memory)
import os
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, auc, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
from config import get_args
from movielens import MovieLens1MDataset
from models.dcnv3 import DCNv3
from models.deepfm import DeepFM
from models.mlp import MLP

def main():
    args = get_args()
    device = torch.device(args.device)
    pin_memory = True if device.type == 'cuda' else False
    
    # Load dataset
    dataset = MovieLens1MDataset(args.dataset_path)
    num_users = len(dataset.user_ids())
    num_items = len(dataset.movie_ids())
    
    # Replicate the split with same seed
    indices = np.arange(len(dataset))
    train_indices, temp_indices = train_test_split(indices, test_size=1 - args.train_ratio, random_state=args.seed)
    val_indices, test_indices = train_test_split(temp_indices, test_size=(1 - args.train_ratio - args.val_ratio) / (1 - args.train_ratio), random_state=args.seed)
    
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    
    # Initialize model
    if args.model_type == 'dcnv3':
        model = DCNv3(
            num_users=num_users, 
            num_items=num_items, 
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
    
    # Load checkpoint
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    
    # Classification Metrics 
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            items, targets = batch
            user_ids, item_ids = items[:, 0], items[:, 1]
            targets_np = targets.float().cpu().numpy()  # Collect on CPU
            
            if args.model_type == 'dcnv3':
                outputs = model(user_ids, item_ids)['y_pred']
            else:
                outputs = model(user_ids, item_ids)
            preds = outputs.cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets_np)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    auc_score = roc_auc_score(all_targets, all_preds)
    logloss = log_loss(all_targets, all_preds)
    accuracy = accuracy_score(all_targets, (all_preds > 0.5).astype(int))
    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    pr_auc = auc(recall, precision)
    cm = confusion_matrix(all_targets, (all_preds > 0.5).astype(int))
    
    print(f'AUC: {auc_score:.4f}')
    print(f'Log Loss: {logloss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'PR-AUC: {pr_auc:.4f}')
    print('Confusion Matrix:')
    print(cm)
    
    # Ranking Metrics (recall@K, precision@K)
    user_train_items = defaultdict(set)
    user_test_pos = defaultdict(set)
    
    for idx in train_indices:
        u, i = dataset.items[idx]
        user_train_items[u].add(i)
    
    for idx in test_indices:
        u, i = dataset.items[idx]
        if dataset.targets[idx] == 1:
            user_test_pos[u].add(i)
    
    precs = []
    recs = []
    
    for user in user_test_pos:
        gt = user_test_pos[user]
        if len(gt) == 0:
            continue
        
        candidates = [it for it in range(num_items) if it not in user_train_items[user]]
        
        if len(candidates) == 0:
            continue
        
        user_tensor = torch.full((len(candidates),), user, dtype=torch.long, device=device)
        item_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        
        with torch.no_grad():
            if args.model_type == 'dcnv3':
                outputs = model(user_tensor, item_tensor)['y_pred']
            else:
                outputs = model(user_tensor, item_tensor)
        
        scores = outputs.cpu().numpy()
        sorted_idx = np.argsort(-scores)
        topk_items = [candidates[j] for j in sorted_idx[:args.topk]]
        
        hit = len(set(topk_items) & gt)
        prec = hit / min(args.topk, len(candidates))
        rec = hit / len(gt)
        
        precs.append(prec)
        recs.append(rec)
    
    if precs:
        mean_prec = np.mean(precs)
        mean_rec = np.mean(recs)
        print(f'Precision@{args.topk}: {mean_prec:.4f}')
        print(f'Recall@{args.topk}: {mean_rec:.4f}')
    else:
        print('No users with positive test items.')
    
    # Visualizations (keeping existing)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_targets, all_preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'roc_curve.png'))
    
    # Precision-Recall Curve
    plt.figure()
    plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'pr_curve.png'))
    
    # Prediction Distribution
    plt.figure()
    plt.hist(all_preds[all_targets == 0], bins=50, alpha=0.5, label='Negative')
    plt.hist(all_preds[all_targets == 1], bins=50, alpha=0.5, label='Positive')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Prediction Distribution')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'pred_dist.png'))

if __name__ == '__main__':

    main()
