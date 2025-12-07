import pandas as pd
import numpy as np
import torch
import os
import joblib # 用于保存和加载预处理器
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

# ==========================================
# 配置部分
# ==========================================
class Config:
    TRAIN_FILE = r'./data/train.csv'
    TEST_FILE = r'./data/test_2.csv'
    
    # 预处理器保存路径
    SCALER_DIR = './checkpoints'
    TABULAR_SCALER_PATH = os.path.join(SCALER_DIR, 'scaler_tabular.pkl')
    SEQ_SCALER_PATH = os.path.join(SCALER_DIR, 'scaler_seq.pkl')
    TARGET_SCALER_PATH = os.path.join(SCALER_DIR, 'scaler_target.pkl')
    
    # 定义列名分组
    FEATURE_COLS = [f'Feature_{i}' for i in range(1, 26)] + ['Ret_MinusTwo', 'Ret_MinusOne']
    SEQ_COLS = [f'Ret_{i}' for i in range(2, 121)]
    
    # 预测目标
    TARGET_INTRADAY = [f'Ret_{i}' for i in range(121, 181)]
    TARGET_DAILY = ['Ret_PlusOne', 'Ret_PlusTwo']
    ALL_TARGETS = TARGET_INTRADAY + TARGET_DAILY

# ==========================================
# 数据集类定义 (保持接口不变)
# ==========================================
class WintonDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str = 'train'):
        self.mode = mode
        
        # 1. 权重处理
        if 'Weight_Intraday' in df.columns:
            self.weight_intraday = df['Weight_Intraday'].values.astype(np.float32)
        else:
            self.weight_intraday = np.ones(len(df), dtype=np.float32)

        if 'Weight_Daily' in df.columns:
            self.weight_daily = df['Weight_Daily'].values.astype(np.float32)
        else:
            self.weight_daily = np.ones(len(df), dtype=np.float32)

        # 2. 静态特征 (已在外部进行 RankGauss 处理)
        self.tabular_data = df[Config.FEATURE_COLS].fillna(0).values.astype(np.float32)

        # 3. 序列特征 (已在外部进行 RobustScaler 处理)
        # [N, 119] -> [N, 119, 1]
        seq_values = df[Config.SEQ_COLS].fillna(0).values.astype(np.float32)
        self.seq_data = seq_values[:, :, np.newaxis]

        # 4. 标签 (已在外部进行 RankGauss 处理)
        if self.mode == 'train':
            self.targets = df[Config.ALL_TARGETS].fillna(0).values.astype(np.float32)
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
# 核心：高阶预处理逻辑
# ==========================================
def preprocess_features(df, is_train=True):
    """
    实现第4名方案的核心预处理：
    1. 填补缺失值
    2. Tabular: RankGauss (QuantileTransformer normal) + Clip
    3. Sequence: RobustScaler + Clip
    4. Target: RankGauss (训练时)
    """
    if not os.path.exists(Config.SCALER_DIR):
        os.makedirs(Config.SCALER_DIR)

    # --- 1. 缺失值处理 ---
    # Feature 列填中位数 (Reference 填的是常数，但中位数更稳健)
    for col in Config.FEATURE_COLS:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    
    # Return 列填 0
    cols_to_zero = Config.SEQ_COLS
    if is_train:
        cols_to_zero = cols_to_zero + Config.ALL_TARGETS
    df[cols_to_zero] = df[cols_to_zero].fillna(0)

    # --- 2. Tabular 特征处理 (RankGauss) ---
    if is_train:
        # 强制高斯分布
        qt_tab = QuantileTransformer(n_quantiles=2000, output_distribution='normal', random_state=42)
        df[Config.FEATURE_COLS] = qt_tab.fit_transform(df[Config.FEATURE_COLS])
        joblib.dump(qt_tab, Config.TABULAR_SCALER_PATH)
    else:
        qt_tab = joblib.load(Config.TABULAR_SCALER_PATH)
        df[Config.FEATURE_COLS] = qt_tab.transform(df[Config.FEATURE_COLS])
    
    # 截断极端值 (Clip at 3 sigma)
    df[Config.FEATURE_COLS] = df[Config.FEATURE_COLS].clip(-3.0, 3.0)

    # --- 3. Sequence 特征处理 (RobustScaler) ---
    # 序列数据不建议用 QuantileTransformer，会破坏波形
    # 使用 RobustScaler 抗异常值
    if is_train:
        scaler_seq = RobustScaler(quantile_range=(5, 95))
        df[Config.SEQ_COLS] = scaler_seq.fit_transform(df[Config.SEQ_COLS])
        joblib.dump(scaler_seq, Config.SEQ_SCALER_PATH)
    else:
        scaler_seq = joblib.load(Config.SEQ_SCALER_PATH)
        df[Config.SEQ_COLS] = scaler_seq.transform(df[Config.SEQ_COLS])
    
    # 截断
    df[Config.SEQ_COLS] = df[Config.SEQ_COLS].clip(-3.0, 3.0)

    # --- 4. Target 处理 (关键！) ---
    # 对 Target 进行高斯化，能显著提升模型训练效果
    # 注意：Inference 时需要反变换！
    if is_train:
        scaler_target = StandardScaler() 
        df[Config.ALL_TARGETS] = scaler_target.fit_transform(df[Config.ALL_TARGETS])
        joblib.dump(scaler_target, Config.TARGET_SCALER_PATH)
    
    return df

def load_and_preprocess_data(filepath, is_train=True):
    print(f"[{'Train' if is_train else 'Test'}] 读取数据: {filepath} ...")
    df = pd.read_csv(filepath) 
    
    # 应用核心预处理
    df = preprocess_features(df, is_train=is_train)
    
    return df

def get_dataloaders(batch_size=64, val_split=0.1):
    # 1. 读取并全量预处理 (Fit & Transform)
    full_df = load_and_preprocess_data(Config.TRAIN_FILE, is_train=True)

    # 2. 划分训练集和验证集
    train_df, val_df = train_test_split(full_df, test_size=val_split, random_state=42)
    
    # 3. 构建 Dataset
    train_dataset = WintonDataset(train_df, mode='train')
    val_dataset = WintonDataset(val_df, mode='train')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"数据加载完成: Train {len(train_dataset)}, Val {len(val_dataset)}")
    return train_loader, val_loader

def get_test_loader(batch_size=64):
    # 读取并预处理 (Transform only using saved scalers)
    test_df = load_and_preprocess_data(Config.TEST_FILE, is_train=False)
    
    test_dataset = WintonDataset(test_df, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader

# ==========================================
# 单元测试
# ==========================================
if __name__ == "__main__":
    # 模拟运行
    train_loader, val_loader = get_dataloaders(batch_size=4)
    print("\n--- 检查数据统计特征 (应接近 N(0,1)) ---")
    for batch in train_loader:
        print(f"Tabular Mean: {batch['tabular'].mean():.4f}, Std: {batch['tabular'].std():.4f}")
        print(f"Target Mean: {batch['target'].mean():.4f}, Std: {batch['target'].std():.4f}")
        break