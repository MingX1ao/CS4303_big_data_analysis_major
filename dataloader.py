import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 配置部分
# ==========================================
class Config:
    TRAIN_FILE = r'./data/train.csv' # 请确保路径正确
    TEST_FILE = r'./data/test_2.csv'
    
    # 定义列名分组
    # 1. 静态特征
    FEATURE_COLS = [f'Feature_{i}' for i in range(1, 26)] + ['Ret_MinusTwo', 'Ret_MinusOne']
    
    # 2. 时间序列特征
    SEQ_COLS = [f'Ret_{i}' for i in range(2, 121)]
    
    # 3. 预测目标
    TARGET_INTRADAY = [f'Ret_{i}' for i in range(121, 181)]
    TARGET_DAILY = ['Ret_PlusOne', 'Ret_PlusTwo']
    ALL_TARGETS = TARGET_INTRADAY + TARGET_DAILY

# ==========================================
# 数据集类定义
# ==========================================
class WintonDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str = 'train'):
        """
        Args:
            df : 预处理后的DataFrame
            mode : 'train' 或 'test'。
        """
        self.mode = mode
        
        # 提取权重
        if 'Weight_Intraday' in df.columns:
            self.weight_intraday = df['Weight_Intraday'].values.astype(np.float32)
        else:
            self.weight_intraday = np.ones(len(df), dtype=np.float32)

        if 'Weight_Daily' in df.columns:
            self.weight_daily = df['Weight_Daily'].values.astype(np.float32)
        else:
            self.weight_daily = np.ones(len(df), dtype=np.float32)

        # 准备静态特征
        self.tabular_data = df[Config.FEATURE_COLS].fillna(0).values.astype(np.float32)

        # 准备序列特征
        seq_values = df[Config.SEQ_COLS].fillna(0).values.astype(np.float32)
        self.seq_data = seq_values[:, :, np.newaxis]        # [N, 119, 1]

        # 准备标签
        if self.mode == 'train':
            self.targets = df[Config.ALL_TARGETS].fillna(0).values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, idx):
        """
        返回一个字典，包含模型需要的所有输入
        """
        sample = {
            'tabular': torch.tensor(self.tabular_data[idx]), # [27]
            'sequence': torch.tensor(self.seq_data[idx]),    # [119, 1]
            'weight_intraday': torch.tensor(self.weight_intraday[idx]),        # scalar
            'weight_daily': torch.tensor(self.weight_daily[idx])        # scalar
        }

        if self.mode == 'train':
            sample['target'] = torch.tensor(self.targets[idx]) # [62] (60分钟 + 2天)

        return sample

# ==========================================
# 数据加载与预处理辅助函数
# ==========================================
def load_and_preprocess_data(filepath, is_train=True):
    """
    这里进行所有的数据清洗工作：缺失值填充、归一化等
    """
    print(f"正在读取数据: {filepath} ...")
    df = pd.read_csv(filepath) 
    
    # 缺失值处理
    for col in Config.FEATURE_COLS:
        df.fillna({col: df[col].median()}, inplace=True)
    for col in Config.SEQ_COLS + Config.ALL_TARGETS:
        df.fillna({col: 0}, inplace=True)
    
    # 归一化
    scaler = StandardScaler()
    df[Config.FEATURE_COLS] = scaler.fit_transform(df[Config.FEATURE_COLS])
    df[Config.SEQ_COLS] = scaler.fit_transform(df[Config.SEQ_COLS])
    
    return df

def get_dataloaders(batch_size=64, val_split=0.1):
    """
    主入口函数：被 train.py 调用
    返回: train_loader, val_loader
    """
    # 读取并清洗训练数据
    full_df = load_and_preprocess_data(Config.TRAIN_FILE, is_train=True)

    # 划分训练集和验证集
    train_df, val_df = train_test_split(full_df, test_size=val_split, random_state=42)
    
    # 构建 Dataset 对象
    train_dataset = WintonDataset(train_df, mode='train')
    val_dataset = WintonDataset(val_df, mode='train')

    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"数据加载完成: Train集 {len(train_dataset)} 样本, Val集 {len(val_dataset)} 样本")
    
    return train_loader, val_loader

def get_test_loader(batch_size=64):
    """
    用于 inference.py 调用
    """
    test_df = load_and_preprocess_data(Config.TEST_FILE, is_train=False)
    test_dataset = WintonDataset(test_df, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=4)
    
    print("\n--- 测试 Batch 数据结构 ---")
    for batch in train_loader:
        print("Tabular shape:", batch['tabular'].shape)   # 预期: [4, 27]
        print("Sequence shape:", batch['sequence'].shape) # 预期: [4, 119, 1]
        print("Target shape:", batch['target'].shape)     # 预期: [4, 62]
        print("Weight intraday shape:", batch['weight_intraday'].shape)     # 预期: [4]
        print("Weight daily shape:", batch['weight_daily'].shape)     # 预期: [4]
        break