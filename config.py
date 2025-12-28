import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Recommendation System Training/Evaluation')
    
    # General
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'], help='Mode: train or evaluate')
    parser.add_argument('--model_type', type=str, default='dcnv3', choices=['dcnv3', 'deepfm', 'mlp'], help='Model architecture')
    parser.add_argument('--dataset_path', type=str, default='ratings.dat', help='Path to MovieLens 1M ratings file')
    parser.add_argument('--users_path', type=str, default='users.dat', help='Path to users.dat')
    parser.add_argument('--movies_path', type=str, default='movies.dat', help='Path to movies.dat')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/model.pth', help='Path to save/load model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Dataset and Splitting
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio for training split')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio for validation split (test will be 1 - train - val)')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for data loaders')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    # Model Hyperparameters 
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension for users and items')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (used differently per model)')
    
    # DCNv3 
    parser.add_argument('--num_deep_cross_layers', type=int, default=2, help='Number of deep cross layers in DCNv3')
    parser.add_argument('--num_shallow_cross_layers', type=int, default=5, help='Number of shallow cross layers in DCNv3')
    parser.add_argument('--deep_net_dropout', type=float, default=0, help='Dropout for deep net in DCNv3')
    parser.add_argument('--shallow_net_dropout', type=float, default=0, help='Dropout for shallow net in DCNv3')
    parser.add_argument('--layer_norm', action='store_true', help='Use layer norm in DCNv3')
    parser.add_argument('--batch_norm', action='store_true', help='Use batch norm in DCNv3')
    parser.add_argument('--loss_type', type=str, default='weighted_tribce', choices=['tribce', 'weighted_tribce', 'bce', 'weighted_bce'], help='Loss type (tribce variants for DCNv3, others for MLP/DeepFM)')
    
    # DeepFM 
    parser.add_argument('--deepfm_hidden_layers', type=str, default='128,64', help='Comma-separated hidden layers for DeepFM DNN (e.g., 128,64)')
    
    # MLP 
    parser.add_argument('--mlp_hidden_layers', type=str, default='64,32', help='Comma-separated hidden layers for MLP (e.g., 128,64,32)')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--early_stop_patience', type=int, default=8, help='Patience for early stopping')
    
    # Evaluation
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save evaluation results and visualizations')
    parser.add_argument('--topk', type=int, default=5, help='K for recall@K and precision@K')
    
    return parser.parse_args()




