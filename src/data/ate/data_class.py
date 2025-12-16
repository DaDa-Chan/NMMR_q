import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "right_heart_catheterization"
ALL_FEATURE_FILE = "RHC_X_allfeatures_list.csv"
SIG_FEATURE_FILE = "RHC_X_significantfeatures_list.csv"
TREATMENT_COL = "swang1"
OUTCOME_COL = "t3d30"
Z_COLS = ["pafi1", "paco21"]
W_COLS = ["ph1", "hema1"]


class RHCDataset(Dataset):
    """
    直接读取 RHC CSV 切分并转换为 PyTorch 张量的 Dataset。
    原先 rhc_experiment.py 中关于 train/val/test 的构造逻辑已经全部挪入此处。
    """

    def __init__(
        self,
        split: str = "train",
        use_all_x: bool = False,
        data_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.split = split.lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}', expected 'train', 'val' or 'test'.")

        self.use_all_x = use_all_x
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.device = device
        self.dtype = dtype

        df = self._load_split_dataframe()
        feature_names = self._load_feature_names()

        self.backdoor = torch.tensor(df[feature_names].values, dtype=dtype, device=device)
        self.treatment = torch.tensor(df[TREATMENT_COL].values, dtype=dtype, device=device).unsqueeze(-1)
        self.outcome = torch.tensor(df[OUTCOME_COL].values, dtype=dtype, device=device).unsqueeze(-1)
        self.treatment_proxy = torch.tensor(df[Z_COLS].values, dtype=dtype, device=device)
        self.outcome_proxy = torch.tensor(df[W_COLS].values, dtype=dtype, device=device)
        self._ate_treatment = torch.tensor([[0.0], [1.0]], dtype=dtype, device=device)

        # 为了与 SGD 数据集共用训练/评估管线，提供统一的属性别名
        self.A = self.treatment
        self.W = self.outcome_proxy
        self.Z = self.treatment_proxy
        self.X = self.backdoor
        self.Y = self.outcome

    def _load_split_dataframe(self) -> pd.DataFrame:
        split_path = self.data_dir / f"rhc_{self.split}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        return pd.read_csv(split_path)

    def _load_feature_names(self) -> list[str]:
        feature_file = ALL_FEATURE_FILE if self.use_all_x else SIG_FEATURE_FILE
        feature_path = self.data_dir / feature_file
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature list not found: {feature_path}")
        return pd.read_csv(feature_path)["variable"].tolist()

    def __len__(self):
        return self.treatment.shape[0]

    def __getitem__(self, idx):
        return {
            'A': self.A[idx],
            'W': self.W[idx],
            'Z': self.Z[idx],
            'X': self.X[idx],
            'Y': self.Y[idx],
        }

    def get_ate_payload(self):
        """
        replicate generate_test_rhc 输出：返回 treatment grid、outcome proxy 与 backdoor。
        """
        return {
            "treatment": self._ate_treatment.clone(),
            "outcome_proxy": self.outcome_proxy,
            "backdoor": self.backdoor,
        }


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

class MergedDataset(Dataset):
    """
    将多个 Dataset 对象（需具有 A, W, Z, X, Y 属性）合并为一个大的 Dataset。
    用于解决 ConcatDataset 无法暴露 .A, .X 等属性给 _build_dataset_view 使用的问题。
    """
    def __init__(self, datasets: List[Dataset]):
        # 假设所有 dataset 都在同一设备上，直接拼接 Tensor
        self.A = torch.cat([d.A for d in datasets], dim=0)
        self.W = torch.cat([d.W for d in datasets], dim=0)
        self.Z = torch.cat([d.Z for d in datasets], dim=0)
        self.X = torch.cat([d.X for d in datasets], dim=0)
        self.Y = torch.cat([d.Y for d in datasets], dim=0)
        
        # RHC 数据集通常没有 U，SGD 有。如果有 U 则拼接，没有则置 None
        if hasattr(datasets[0], 'U') and datasets[0].U is not None:
            self.U = torch.cat([d.U for d in datasets], dim=0)
        else:
            self.U = None

    def __len__(self):
        return self.A.shape[0]

    def __getitem__(self, idx):
        item = {
            'A': self.A[idx],
            'W': self.W[idx],
            'Z': self.Z[idx],
            'X': self.X[idx],
            'Y': self.Y[idx],
        }
        if self.U is not None:
            item['U'] = self.U[idx]
        return item