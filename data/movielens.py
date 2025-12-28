import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

class MovieLens20MDataset(Dataset):
    def __init__(self, dataset_path, sep=',', engine='c', header='infer'):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
        raw_users = data[:, 0].astype(np.int64)
        raw_items = data[:, 1].astype(np.int64)
        
        # Safely remap to contiguous 0-based indices
        unique_users = np.unique(raw_users)
        unique_items = np.unique(raw_items)
        user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        item_map = {iid: idx for idx, iid in enumerate(unique_items)}
        
        self.items = np.stack([np.vectorize(user_map.get)(raw_users),
                               np.vectorize(item_map.get)(raw_items)], axis=1)
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        self.field_dims = np.array([len(unique_users), len(unique_items)], dtype=np.int64)
        self.user_field_idx = np.array((0,), dtype=np.int64)
        self.item_field_idx = np.array((1,), dtype=np.int64)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target

    def targets_count(self):
        unique, counts = np.unique(self.targets, return_counts=True)
        return dict(zip(unique, counts))

    def user_ids(self): return np.unique(self.items[:, 0])
    def movie_ids(self): return np.unique(self.items[:, 1])
    def user_id_column(self): return self.items[:, 0]
    def movie_id_column(self): return self.items[:, 1]

    @property
    def shape(self): return self.items.shape

class MovieLens1MDataset(MovieLens20MDataset):
    def __init__(self, dataset_path):
        super().__init__(dataset_path, sep='::', engine='python', header=None)

class MovieLens1MDatasetWithMetadata(Dataset):
    def __init__(self, ratings_path, users_path='users.dat', movies_path='movies.dat'):
        ratings_df = pd.read_csv(ratings_path, sep='::', engine='python', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
        
        # Remap IDs
        self.user_map = {uid: idx for idx, uid in enumerate(ratings_df['user_id'].unique())}
        self.movie_map = {mid: idx for idx, mid in enumerate(ratings_df['movie_id'].unique())}
        ratings_df['user_idx'] = ratings_df['user_id'].map(self.user_map)
        ratings_df['movie_idx'] = ratings_df['movie_id'].map(self.movie_map)
        
        self.user_ids = ratings_df['user_idx'].values.astype(np.int64)
        self.movie_ids = ratings_df['movie_idx'].values.astype(np.int64)
        self.targets = (ratings_df['rating'] > 3).astype(np.float32)
        
        # Users metadata
        users_df = pd.read_csv(users_path, sep='::', engine='python', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zip'])
        users_df['age_bin'] = pd.cut(users_df['age'], bins=[1,18,25,35,45,50,56,100], labels=range(7))
        users_df['gender'] = users_df['gender'].map({'M': 0, 'F': 1}).fillna(0)
        self.user_meta = {self.user_map[row['user_id']]: (int(row['age_bin']), int(row['gender']), int(row['occupation'])) for _, row in users_df.iterrows() if row['user_id'] in self.user_map}
        
        # Movies metadata (genres multi-hot)
        movies_df = pd.read_csv(movies_path, sep='::', engine='python', header=None, names=['movie_id', 'title', 'genres'])
        self.genre_list = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.movie_meta = {}
        for _, row in movies_df.iterrows():
            if row['movie_id'] in self.movie_map:
                genres = np.zeros(len(self.genre_list), dtype=np.float32)
                for g in str(row['genres']).split('|'):
                    if g in self.genre_list:
                        genres[self.genre_list.index(g)] = 1.0
                self.movie_meta[self.movie_map[row['movie_id']]] = genres
        
        # Field dims (sparse only)
        self.field_dims = np.array([len(self.user_map), len(self.movie_map), 7, 2, 21], dtype=np.int64)  # user, movie, age_bin, gender, occupation
        self.dense_dim = len(self.genre_list)  # genres

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        u_id = self.user_ids[index]
        m_id = self.movie_ids[index]
        user_m = self.user_meta.get(u_id, (0, 0, 0))
        sparse = torch.tensor([u_id, m_id, user_m[0], user_m[1], user_m[2]], dtype=torch.long)
        dense = torch.tensor(self.movie_meta.get(m_id, np.zeros(self.dense_dim, dtype=np.float32)), dtype=torch.float32)
        return sparse, dense, torch.tensor(self.targets[index], dtype=torch.float32)
