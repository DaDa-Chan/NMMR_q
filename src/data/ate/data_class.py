import torch
from torch.utils.data import Dataset
import pandas as pd


class SGDDataset(Dataset):
    def __init__(self, csv_path, device='cpu', dtype=torch.float32, df: pd.DataFrame | None = None):
        if df is not None:
            self.data = df
        else:
            self.data = pd.read_csv(csv_path)
        self.X = torch.tensor(self.data[['X1', 'X2']].values, dtype=dtype, device=device)
        self.A = torch.tensor(self.data['A'].values, dtype=dtype, device=device).unsqueeze(1)
        self.Z = torch.tensor(self.data['Z'].values, dtype=dtype, device=device).unsqueeze(1)
        self.W = torch.tensor(self.data['W'].values, dtype=dtype, device=device).unsqueeze(1)
        self.Y = torch.tensor(self.data['Y'].values, dtype=dtype, device=device).unsqueeze(1)
        if 'U' in self.data.columns:
            self.U = torch.tensor(self.data['U'].values, dtype=torch.float32).unsqueeze(1).to(device)
        else:
            self.U = torch.zeros_like(self.A)
            
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'A': self.A[idx],
            'Z': self.Z[idx],
            'W': self.W[idx],
            'U': self.U[idx],
            'Y': self.Y[idx],
        }
