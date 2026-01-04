import torch
import numpy as np
from movielens import MovieLens1MDatasetWithMetadata
from models.dcnv3 import DCNv3

def predict(user_id):
    # Load dataset and model
    dataset = MovieLens1MDatasetWithMetadata('ratings.dat', 'users.dat', 'movies.dat')
    field_dims = dataset.field_dims
    dense_dim = dataset.dense_dim
    model = DCNv3(
        field_dims=field_dims,
        dense_dim=dense_dim,
        embed_dim=32,
        num_deep_cross_layers=5,
        num_shallow_cross_layers=2,
        deep_net_dropout=0,
        shallow_net_dropout=0,
        layer_norm=False,
        batch_norm=False
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load('./checkpoints/dcnv3.pt', map_location=device))
    model.eval()

    # Get interacted movies (internal IDs)
    user_mask = dataset.items[:, 0] == user_id
    interacted = set(dataset.items[user_mask, 1])

    # Candidates: non-interacted movies
    num_items = len(dataset.movie_ids())
    candidates = [i for i in range(num_items) if i not in interacted]
    if not candidates:
        return {}

    # Build sparse and dense inputs
    user_tensor = torch.full((len(candidates),), user_id, dtype=torch.long, device=device)
    item_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
    user_m = dataset.user_meta.get(user_id, (0, 0, 0))
    age_bin = torch.full((len(candidates),), user_m[0], dtype=torch.long, device=device)
    gender = torch.full((len(candidates),), user_m[1], dtype=torch.long, device=device)
    occ = torch.full((len(candidates),), user_m[2], dtype=torch.long, device=device)
    sparse = torch.stack([user_tensor, item_tensor, age_bin, gender, occ], dim=1)

    dense_list = [torch.tensor(dataset.movie_meta.get(c, np.zeros(dense_dim, dtype=np.float32))) for c in candidates]
    dense = torch.stack(dense_list, dim=0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(sparse, dense)['y_pred']
    scores = outputs.cpu().numpy()

    # Sort descending and build dict
    sorted_idx = np.argsort(-scores)
    result = {candidates[idx]: float(scores[idx]) for idx in sorted_idx}
    return result