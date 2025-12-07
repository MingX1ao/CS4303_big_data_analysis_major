import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from model import WintonBaselineModel
from dataloader import get_test_loader, Config

class InferenceConfig:
    MODEL_PATH = './checkpoints/winton_hybrid_v1.pth'
    # 修改输出后缀为 .csv.gz，Pandas会自动压缩，生成的 zip/gz 文件可以直接提交
    # 这样可以避免生成几百MB的大文件
    OUTPUT_FILE = 'submission.csv.gz' 
    BATCH_SIZE = 256
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    ENABLE_ZERO_MEAN = True

def predict_all(model, loader, device):
    model.eval()
    all_preds = []
    print(f"开始推理 (Device: {device})...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            x_tab = batch['tabular'].to(device)
            x_seq = batch['sequence'].to(device)
            preds = model(x_seq, x_tab)
            all_preds.append(preds.cpu().numpy())
    return np.vstack(all_preds)

def generate_submission(predictions):
    """
    使用 NumPy 高效生成符合 1_1, 1_2 ... 1_62, 2_1 ... 顺序的提交文件
    """
    print("正在生成提交文件结构...")
    
    # 1. 获取测试集 ID
    test_df = pd.read_csv(Config.TEST_FILE, usecols=['Id'])
    test_ids = test_df['Id'].values
    n_samples = len(test_ids)
    
    # 检查样本数量
    # Winton 完整测试集应该是 120,000 行
    # 如果你用的是子集，行数会不同，但逻辑依然成立
    assert n_samples == predictions.shape[0], "样本数与预测数不匹配"
    
    # 2. 构造所有的 ID 字符串
    # 目标格式: Id_Suffix
    # 我们需要将 test_ids 重复 62 次: [1, 1...1, 2, 2...2]
    # 我们需要将 1-62 循环 n 次: [1, 2...62, 1, 2...62]
    
    # 步骤 A: 准备 ID 列 (Repeat) -> [1, 1, ..., 2, 2, ...]
    # 注意：Run-length repeat
    ids_repeated = np.repeat(test_ids, 62)
    
    # 步骤 B: 准备后缀列 (Tile) -> [1, 2, ..., 62, 1, 2, ..., 62]
    suffixes = np.tile(np.arange(1, 63), n_samples)
    
    # 步骤 C: 拼接字符串 (这一步比较耗时，但这是生成 CSV 必须的)
    # 为了加速，我们先构造 DataFrame，利用 Pandas 的向量化操作
    # 这里的顺序自然就是 Row-Major 的
    
    print(f"构造 DataFrame (Total Rows: {len(ids_repeated)})...")
    submission = pd.DataFrame({
        'Id_Num': ids_repeated,
        'Suffix': suffixes,
        # flatten() 默认是按行展平，正好对应 1_1, 1_2, ..., 1_62, 2_1 ...
        'Predicted': predictions.flatten() 
    })
    
    # 3. 组合最终 ID
    # 格式: {Id_Num}_{Suffix}
    submission['Id'] = submission['Id_Num'].astype(str) + '_' + submission['Suffix'].astype(str)
    
    # 4. 只保留需要的列
    submission = submission[['Id', 'Predicted']]
    
    return submission

def main():
    print(f"Using device: {InferenceConfig.DEVICE}")
    
    test_loader = get_test_loader(batch_size=InferenceConfig.BATCH_SIZE)
    model = WintonBaselineModel().to(InferenceConfig.DEVICE)
    
    if os.path.exists(InferenceConfig.MODEL_PATH):
        model.load_state_dict(torch.load(InferenceConfig.MODEL_PATH, map_location=InferenceConfig.DEVICE))
    else:
        raise FileNotFoundError(f"找不到模型文件: {InferenceConfig.MODEL_PATH}")

    # 预测
    raw_preds = predict_all(model, test_loader, InferenceConfig.DEVICE)
    
    # 后处理
    if InferenceConfig.ENABLE_ZERO_MEAN:
        print("应用 Zero-Mean 修正...")
        raw_preds = raw_preds - np.mean(raw_preds, axis=0)
        
    # 生成提交
    submission_df = generate_submission(raw_preds)
    
    # 验证行数 (如果是完整测试集，应为 7,440,000)
    print(f"生成的行数: {len(submission_df)}")
    
    # 保存 (自动压缩)
    print(f"正在保存到 {InferenceConfig.OUTPUT_FILE} ...")
    submission_df.to_csv(InferenceConfig.OUTPUT_FILE, index=False, compression='gzip')
    print("✅ 完成！")
    print(submission_df.head())

if __name__ == "__main__":
    main()