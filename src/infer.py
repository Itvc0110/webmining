import torch
import numpy as np
from src.data_utils import load_dataset
from src.model_loader import load_model_dcnv3

def dcnv3_predict(model, dataset, user_id, top_k=5):
    """
    Predict top-k movies for a given user (internal user_idx)
    """
    device = next(model.parameters()).device

    # =====================
    # 1. Movies user already interacted
    # =====================
    mask = dataset.user_ids == user_id
    interacted_movie_idxs = set(dataset.movie_ids[mask].tolist())

    # =====================
    # 2. Candidate movies (not seen)
    # =====================
    all_movie_idxs = np.unique(dataset.movie_ids)
    candidates = [mid for mid in all_movie_idxs if mid not in interacted_movie_idxs]

    if len(candidates) == 0:
        return {}

    # =====================
    # 3. Build sparse features
    # order MUST match dataset.field_dims
    # [user, movie, age, gender, occupation]
    # =====================
    num_candidates = len(candidates)

    user_tensor = torch.full(
        (num_candidates,), user_id, dtype=torch.long, device=device
    )
    movie_tensor = torch.tensor(
        candidates, dtype=torch.long, device=device
    )

    age, gender, occ = dataset.user_meta.get(user_id, (0, 0, 0))
    age_tensor = torch.full((num_candidates,), age, dtype=torch.long, device=device)
    gender_tensor = torch.full((num_candidates,), gender, dtype=torch.long, device=device)
    occ_tensor = torch.full((num_candidates,), occ, dtype=torch.long, device=device)

    sparse = torch.stack(
        [user_tensor, movie_tensor, age_tensor, gender_tensor, occ_tensor],
        dim=1
    )

    # =====================
    # 4. Dense movie features
    # =====================
    dense = torch.stack(
        [
            torch.tensor(
                dataset.movie_meta.get(
                    mid,
                    np.zeros(dataset.dense_dim, dtype=np.float32)
                ),
                dtype=torch.float32
            )
            for mid in candidates
        ],
        dim=0
    ).to(device)

    # =====================
    # 5. Predict
    # =====================
    with torch.no_grad():
        output = model(sparse, dense)

        # DCNv3 returns dict
        scores = output["y_pred"].view(-1).cpu().numpy()

    # =====================
    # 6. Top-k
    # =====================
    top_idx = np.argsort(-scores)[:top_k]

    result = {
        int(candidates[i]): float(scores[i])
        for i in top_idx
    }

    return result



# =====================
# TEST
# =====================
if __name__ == "__main__":
    print("🔹 Loading dataset...")
    dataset = load_dataset()
    model, device  = load_model_dcnv3(
        dataset,
        r"src\checkpoints\dcnv3.pth",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Finished laoding model and dataset")
    
    for user_id in [1, 2]:
        print(f"\n🚀 Testing DCNv3 for user_id = {user_id}")

        result = dcnv3_predict(model, dataset, user_id, top_k=5)

        if not result:
            print("⚠️ No recommendation (user watched all movies?)")
            continue

        for rank, (mid, score) in enumerate(result.items(), 1):
            title = dataset.movie_titles.get(mid, "Unknown")
            print(f"{rank}. {title} (movie_id={mid}) → score={score:.4f}")
