# dataset class

import torch
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    """
    input is already scaled (normalised), already split to split_type, need to convert to Tensor.
    """
    def __init__(self, X, y):        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]

# # sample code
# train_dataset = StockDataset(X=X_train,y=y_train)
# val_dataset = StockDataset(X=X_val,y=y_val)
# test_dataset = StockDataset(X=X_test,y=y_test)

# print(train_dataset.X.shape,val_dataset.X.shape,test_dataset.X.shape)
# print(train_dataset.y.shape,val_dataset.y.shape,test_dataset.y.shape)