import numpy as np
import pandas as pd
from src.movielens import MovieLens1MDatasetWithMetadata

def load_dataset(
    ratings_path=r'data_source\ml-1m\ratings.dat',
    users_path=r'data_source\ml-1m\users.dat',
    movies_path=r'data_source\ml-1m\movies.dat'
):
    dataset = MovieLens1MDatasetWithMetadata(
        ratings_path=ratings_path, 
        users_path=users_path, 
        movies_path=movies_path,
    )
    return dataset


def get_all_watched_movies(dataset, user_idx):
    user_idx = int(user_idx)
    mask = dataset.user_ids == user_idx
    movie_idxs = dataset.movie_ids[mask]
    ratings = dataset.ratings[mask]

    print(f"User {user_idx} has watched {len(movie_idxs)} movies.")
    
    rows = []
    for mid, r in zip(movie_idxs, ratings):
        rows.append({
            "Movie ID": int(mid),
            "Title": dataset.movie_titles.get(int(mid), "Unknown"),
            "Rating": float(r)
        })
        
    return pd.DataFrame(rows)

def get_top_n_movies(dataset, user_idx, n=5):
    user_idx = int(user_idx)

    mask = dataset.user_ids == user_idx

    movie_ids = dataset.movie_ids[mask]
    ratings = dataset.ratings[mask]

    pairs = sorted(zip(movie_ids, ratings), key=lambda x: x[1], reverse=True)

    seen = set()
    rows = []

    for mid, rating in pairs:
        if mid in seen:
            continue
        seen.add(mid)

        rows.append({
            "Movie ID": int(mid),
            "Title": dataset.movie_titles.get(int(mid), "Unknown"),
            "Rating": float(rating)
        })

        if len(rows) == n:
            break

    return pd.DataFrame(rows)

if __name__ == "__main__":
    dataset = load_dataset()
    user_id = 2

    watched = get_all_watched_movies(dataset, user_id)
    top5 = get_top_n_movies(dataset, user_id, n=5)

    print("Watched:", len(watched))
    print("Top 5:", top5)
