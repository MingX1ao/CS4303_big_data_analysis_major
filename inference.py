import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# 引入项目模块
from model import WintonBaselineModel
from dataloader import get_test_loader, Config

# ==========================================
# 配置
# ==========================================
class InferenceConfig:
    MODEL_PATH = './checkpoints/winton_hybrid_v1.pth' # 请确保路径和文件名正确
    OUTPUT_FILE = 'submission.csv'
    BATCH_SIZE = 256     # 推理时Batch可以大一点
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 结果修正技巧
    ENABLE_ZERO_MEAN = True  # 强制将预测结果的列均值设为0

# ==========================================
# 推理函数
# ==========================================
def predict_all(model, loader, device):
    model.eval()
    all_preds = []
    
    print(f"开始推理 (Device: {device})...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            x_tab = batch['tabular'].to(device)
            x_seq = batch['sequence'].to(device)
            
            # 模型预测 [Batch, 62]
            preds = model(x_seq, x_tab)
            
            # 将结果转回 CPU 并存入列表
            all_preds.append(preds.cpu().numpy())
            
    # 拼接所有 Batch 的结果 -> Shape: [Total_Test_Samples, 62]
    return np.vstack(all_preds)

def generate_submission(predictions):
    """
    将预测矩阵转换为 Winton Challenge 要求的格式:
    Id, Predicted
    1_1, value
    ...
    1_62, value
    """
    print("正在生成提交文件...")
    
    # 1. 读取测试集的原始 ID
    # Winton 的 test_2.csv 第一列通常是 Id
    test_df = pd.read_csv(Config.TEST_FILE, usecols=['Id'])
    test_ids = test_df['Id'].values
    
    # 检查行数匹配
    assert len(test_ids) == predictions.shape[0], \
        f"ID数量 ({len(test_ids)}) 与 预测数量 ({predictions.shape[0]}) 不匹配！"

    # 2. 构建 DataFrame
    # 列名直接设为字符串 '1' 到 '62'，方便后续 melt 拼接
    # 我们的模型输出顺序就是 Ret_121...Ret_180, PlusOne, PlusTwo
    # 正好对应题目要求的 1...62
    col_names = [str(i) for i in range(1, 63)]
    
    pred_df = pd.DataFrame(predictions, columns=col_names)
    pred_df['Id'] = test_ids # 添加行 ID

    # 3. Melt (宽表变长表)
    # id_vars='Id' -> 保持 Id 列不变
    # var_name='Suffix' -> 原来的列名('1'...'62')变成一列，叫 Suffix
    # value_name='Predicted' -> 值变成一列
    melted = pred_df.melt(id_vars=['Id'], var_name='Suffix', value_name='Predicted')
    
    # 4. 生成最终 ID: {RowId}_{Suffix}
    # 例如: 1 + '_' + 1 -> 1_1
    print("正在拼接 ID 字符串...")
    melted['Id'] = melted['Id'].astype(str) + '_' + melted['Suffix']
    
    # 5. 排序 (可选，为了好看，Kaggle通常只看ID匹配)
    # 为了让提交文件看起来像样例 (1_1, 1_2 ... 1_62, 2_1 ...)，我们需要特殊的排序
    # 因为纯字符串排序 '1_10' 会排在 '1_2' 前面，所以我们先不sort，依赖 melt 的默认顺序
    # melt 默认会把所有行的 '1' 排完，再排 '2'。
    # 如果想严格按照样例顺序 (Row major)，我们可以这样做：
    # 但是对于几十万行数据，只要ID对即可，不用强求顺序。
    # 这里直接输出。
    
    submission = melted[['Id', 'Predicted']]
    return submission

# ==========================================
# 主程序
# ==========================================
def main():
    # 1. 准备环境
    # 2. 加载数据
    test_loader = get_test_loader(batch_size=InferenceConfig.BATCH_SIZE)
    
    # 3. 加载模型
    model = WintonBaselineModel().to(InferenceConfig.DEVICE)
    
    if os.path.exists(InferenceConfig.MODEL_PATH):
        model.load_state_dict(torch.load(InferenceConfig.MODEL_PATH, map_location=InferenceConfig.DEVICE))
        print(f"成功加载模型权重: {InferenceConfig.MODEL_PATH}")
    else:
        raise FileNotFoundError(f"找不到模型文件: {InferenceConfig.MODEL_PATH}")

    # 4. 执行预测
    raw_preds = predict_all(model, test_loader, InferenceConfig.DEVICE)
    
    # 5. 后处理 (Post-processing)
    if InferenceConfig.ENABLE_ZERO_MEAN:
        print("应用 Zero-Mean 修正 (Subtract Column Means)...")
        column_means = np.mean(raw_preds, axis=0)
        raw_preds = raw_preds - column_means
        
    # 6. 格式化与保存
    submission_df = generate_submission(raw_preds)
    
    # 排序优化：虽然melt后的顺序也能提交，但为了保险，
    # 我们可以尝试按照 Id 的整数值排序（如果内存允许）。
    # 对于 Winton 这种大数据量，直接保存通常也没问题。
    
    submission_df.to_csv(InferenceConfig.OUTPUT_FILE, index=False)
    print(f"✅ 提交文件已生成: {InferenceConfig.OUTPUT_FILE}")
    
    # 打印前几行检查格式
    print("\n--- 提交文件预览 ---")
    print(submission_df.head(10))

if __name__ == "__main__":
    main()