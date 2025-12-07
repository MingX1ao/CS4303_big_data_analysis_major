import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==========================================
# 配置部分（保持不变）
# ==========================================
class Config:
    TRAIN_FILE = r'./data/train.csv'
    TEST_FILE = r'./data/test_2.csv'

    FEATURE_COLS = [f'Feature_{i}' for i in range(1, 26)] + ['Ret_MinusTwo', 'Ret_MinusOne']
    SEQ_COLS = [f'Ret_{i}' for i in range(2, 121)]

    TARGET_INTRADAY = [f'Ret_{i}' for i in range(121, 181)]
    TARGET_DAILY = ['Ret_PlusOne', 'Ret_PlusTwo']
    ALL_TARGETS = TARGET_INTRADAY + TARGET_DAILY


# ==========================================
# 数据集定义（接口保持不变）
# ==========================================
class WintonDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str = 'train'):
        self.mode = mode

        if 'Weight_Intraday' in df.columns:
            self.weight_intraday = df['Weight_Intraday'].values.astype(np.float32)
        else:
            self.weight_intraday = np.ones(len(df), dtype=np.float32)

        if 'Weight_Daily' in df.columns:
            self.weight_daily = df['Weight_Daily'].values.astype(np.float32)
        else:
            self.weight_daily = np.ones(len(df), dtype=np.float32)

        self.tabular_data = df[Config.FEATURE_COLS].values.astype(np.float32)

        seq_values = df[Config.SEQ_COLS].values.astype(np.float32)
        self.seq_data = seq_values[:, :, np.newaxis]

        if self.mode == 'train':
            self.targets = df[Config.ALL_TARGETS].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, idx):
        sample = {
            'tabular': torch.tensor(self.tabular_data[idx]),
            'sequence': torch.tensor(self.seq_data[idx]),
            'weight_intraday': torch.tensor(self.weight_intraday[idx]),
            'weight_daily': torch.tensor(self.weight_daily[idx])
        }

        if self.mode == 'train':
            sample['target'] = torch.tensor(self.targets[idx])

        return sample


# ==========================================
# ✅【最终策略】完全无标准化预处理（只做 NaN 处理）
# ==========================================
def preprocess_features(df, is_train=True):
    """
    ✅ 纯原始金融物理尺度版本：
    1. Feature：只填 NaN
    2. Sequence：只填 NaN
    3. Target：只填 NaN
    ❌ 不做任何：
        - StandardScaler
        - QuantileTransformer
        - Clip
        - RankGauss
    """

    # ========= 1. Tabular 特征：NaN → 中位数 =========
    for col in Config.FEATURE_COLS:
        df[col] = df[col].fillna(df[col].median())

    # ========= 2. Sequence：NaN → 0 =========
    df[Config.SEQ_COLS] = df[Config.SEQ_COLS].fillna(0.0)

    # ========= 3. Target：NaN → 0 =========
    if is_train:
        df[Config.ALL_TARGETS] = df[Config.ALL_TARGETS].fillna(0.0)

    return df


def load_and_preprocess_data(filepath, is_train=True):
    print(f"[{'Train' if is_train else 'Test'}] 读取数据: {filepath}")
    df = pd.read_csv(filepath)
    df = preprocess_features(df, is_train=is_train)
    return df


# ==========================================
# Dataloader 接口（保持不变）
# ==========================================
def get_dataloaders(batch_size=64, val_split=0.1):
    full_df = load_and_preprocess_data(Config.TRAIN_FILE, is_train=True)

    train_df, val_df = train_test_split(
        full_df, test_size=val_split, random_state=42
    )

    train_dataset = WintonDataset(train_df, mode='train')
    val_dataset = WintonDataset(val_df, mode='train')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2
    )

    print(f"数据加载完成: Train {len(train_dataset)}, Val {len(val_dataset)}")
    return train_loader, val_loader


def get_test_loader(batch_size=64):
    test_df = load_and_preprocess_data(Config.TEST_FILE, is_train=False)
    test_dataset = WintonDataset(test_df, mode='test')

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2
    )

    return test_loader


# ==========================================
# 单元测试
# ==========================================
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=4)

    print("\n--- 检查数据统计特征（纯原始尺度） ---")
    for batch in train_loader:
        print(f"Tabular Mean: {batch['tabular'].mean():.6f}, Std: {batch['tabular'].std():.6f}")
        print(f"Sequence Mean: {batch['sequence'].mean():.8f}, Std: {batch['sequence'].std():.8f}")
        print(f"Target Mean: {batch['target'].mean():.8f}, Std: {batch['target'].std():.8f}")
        break
