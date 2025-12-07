
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from tqdm import tqdm

from dataloader import get_dataloaders
from model import WintonBaselineModel

# ==========================================
# 配置部分
# ==========================================
class TrainConfig:
    EPOCHS = 50
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_DIR = './checkpoints'
    MODEL_NAME = 'winton_hybrid_v1.pth'
    
    # Daily Loss 更高的系数，强迫模型关注跨日预测
    LAMBDA_INTRADAY = 1.0
    LAMBDA_DAILY = 1.0 

# ==========================================
# 自定义损失函数
# ==========================================
class WintonWeightedLoss(nn.Module):
    def __init__(self):
        super(WintonWeightedLoss, self).__init__()

    def forward(self, preds, targets, w_intra, w_daily):
        """
        preds: [Batch, 62]
        targets: [Batch, 62]
        w_intra: [Batch]
        w_daily: [Batch]
        """
        # 拆分预测值和目标值
        pred_intra = preds[:, :60]
        true_intra = targets[:, :60]
        
        pred_daily = preds[:, 60:]
        true_daily = targets[:, 60:]
        
        # 计算日内部分的 MAE
        # [Batch, 60] -> [Batch] (对每一行求平均误差)
        mae_intra = torch.abs(pred_intra - true_intra).mean(dim=1)
        # 加权: [Batch] * [Batch] -> Mean Scalar
        loss_intra = (mae_intra * w_intra).mean()
        
        # 3. 计算跨日部分的 MAE
        # [Batch, 2] -> [Batch]
        mae_daily = torch.abs(pred_daily - true_daily).mean(dim=1)
        # 加权
        loss_daily = (mae_daily * w_daily).mean()
        
        # 4. 组合总 Loss
        total_loss = (TrainConfig.LAMBDA_INTRADAY * loss_intra) + \
                     (TrainConfig.LAMBDA_DAILY * loss_daily)
                     
        return total_loss, loss_intra.item(), loss_daily.item()

# ==========================================
# 训练与验证函数
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        # 1. 数据搬运到 GPU
        x_tab = batch['tabular'].to(device)
        x_seq = batch['sequence'].to(device)
        targets = batch['target'].to(device)
        w_intra = batch['weight_intraday'].to(device)
        w_daily = batch['weight_daily'].to(device)
        
        # 2. 清零梯度
        optimizer.zero_grad()
        
        # 3. 前向传播
        preds = model(x_seq, x_tab)
        
        # 4. 计算损失
        loss, l_intra, l_daily = criterion(preds, targets, w_intra, w_daily)
        
        # 5. 反向传播与更新
        loss.backward()
        # 梯度裁剪：防止 LSTM 梯度爆炸 (金融数据常见的坑)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        running_loss += loss.item()
        
        # 更新进度条显示
        pbar.set_postfix({'L_Intra': f'{l_intra:.4f}', 'L_Daily': f'{l_daily:.4f}'})

    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_intra = 0.0
    running_daily = 0.0
    
    with torch.no_grad():
        for batch in loader:
            x_tab = batch['tabular'].to(device)
            x_seq = batch['sequence'].to(device)
            targets = batch['target'].to(device)
            w_intra = batch['weight_intraday'].to(device)
            w_daily = batch['weight_daily'].to(device)
            
            preds = model(x_seq, x_tab)
            loss, l_intra, l_daily = criterion(preds, targets, w_intra, w_daily)
            
            running_loss += loss.item()
            running_intra += l_intra
            running_daily += l_daily
            
    # 计算平均 loss
    avg_loss = running_loss / len(loader)
    avg_intra = running_intra / len(loader)
    avg_daily = running_daily / len(loader)
    
    return avg_loss, avg_intra, avg_daily

# ==========================================
# 主程序
# ==========================================
def main():
    # 0. 准备环境
    if not os.path.exists(TrainConfig.SAVE_DIR):
        os.makedirs(TrainConfig.SAVE_DIR)
    
    print(f"Using device: {TrainConfig.DEVICE}")
    
    # 1. 获取数据
    train_loader, val_loader = get_dataloaders(
        batch_size=TrainConfig.BATCH_SIZE, 
        val_split=0.2
    )
    
    # 2. 初始化模型
    model = WintonBaselineModel().to(TrainConfig.DEVICE)
    
    # 3. 定义优化器和损失
    optimizer = optim.AdamW(model.parameters(), lr=TrainConfig.LEARNING_RATE, weight_decay=1e-4)
    # 学习率调度器：如果 Loss 不下降，自动减小学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = WintonWeightedLoss()
    
    # 4. 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(TrainConfig.EPOCHS):
        print(f"\nEpoch {epoch+1}/{TrainConfig.EPOCHS}")
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, TrainConfig.DEVICE)
        
        # 验证
        val_loss, val_intra, val_daily = validate(model, val_loader, criterion, TrainConfig.DEVICE)
        
        # 调度器步进
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.5f}")
        print(f"Val Loss: {val_loss:.5f} (Intra: {val_intra:.5f} | Daily: {val_daily:.5f})")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(TrainConfig.SAVE_DIR, TrainConfig.MODEL_NAME)
            torch.save(model.state_dict(), save_path)
            print(f"--> Best model saved to {save_path}")

if __name__ == "__main__":
    main()