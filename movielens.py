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
