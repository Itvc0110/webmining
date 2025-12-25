import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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

class MovieLens1MDatasetWithMetadata(MovieLens1MDataset):
    def __init__(self, dataset_path, users_path='users.dat', movies_path='movies.dat'):
        super().__init__(dataset_path)
        # Load users
        users_df = pd.read_csv(users_path, sep='::', engine='python', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zip'])
        users_df['user_id'] -= 1  # Zero-index
        users_df['age_bin'] = pd.cut(users_df['age'], bins=[0, 18, 25, 35, 45, 50, 56, 100], labels=range(7), right=False)
        users_df['gender'] = users_df['gender'].map({'M': 0, 'F': 1})
        self.user_meta = {row['user_id']: (int(row['age_bin']), row['gender'], row['occupation']) for _, row in users_df.iterrows()}
        
        # Load movies
        movies_df = pd.read_csv(movies_path, sep='::', engine='python', header=None, names=['movie_id', 'title', 'genres'])
        movies_df['movie_id'] -= 1
        self.genre_list = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.movie_meta = {}
        for _, row in movies_df.iterrows():
            g_vec = np.zeros(len(self.genre_list), dtype=np.float32)
            for g in row['genres'].split('|'):
                if g in self.genre_list:
                    g_vec[self.genre_list.index(g)] = 1.0
            self.movie_meta[row['movie_id']] = g_vec
        
        # Update field_dims
        self.field_dims = np.append(self.field_dims, [7, 2, 21, len(self.genre_list)])  # age_bins, gender, occupation, genres_dim

    def __getitem__(self, index):
        items, target = super().__getitem__(index)
        u, i = items
        user_m = self.user_meta.get(u, (0, 0, 0))  # Default neutral
        movie_m = self.movie_meta.get(i, np.zeros(len(self.genre_list), dtype=np.float32))
        return {
            'user_id': torch.tensor(u, dtype=torch.long),
            'movie_id': torch.tensor(i, dtype=torch.long),
            'age_bin': torch.tensor(user_m[0], dtype=torch.long),
            'gender': torch.tensor(user_m[1], dtype=torch.long),
            'occupation': torch.tensor(user_m[2], dtype=torch.long),
            'genres': torch.tensor(movie_m, dtype=torch.float32)
        }, torch.tensor(target, dtype=torch.float32)
