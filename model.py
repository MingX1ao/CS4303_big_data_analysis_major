import torch
import torch.nn as nn

class WintonBaselineModel(nn.Module):
    def __init__(self, 
                 seq_input_size=1,      # 序列特征维度 (只有1列: return)
                 tabular_input_size=27, # 静态特征维度 (Feature_1~25 + Ret_MinusOne/Two)
                 hidden_size=64,        # LSTM 隐藏层大小
                 output_size=62,        # 输出: 60分钟(121~180) + 2天(D+1, D+2)
                 dropout_prob=0.3):     # Dropout 防止过拟合
        
        super(WintonBaselineModel, self).__init__()

        # ==================================
        # 1. 时序分支 (处理 Ret_2 ~ Ret_120)
        # ==================================
        # 使用 LSTM 提取日内趋势
        # batch_first=True 意味着输入形状是 (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=2,           # 2层 LSTM 稍微增加一点深度
            batch_first=True,
            dropout=dropout_prob
        )
        
        # ==================================
        # 2. 静态分支 (处理 Features)
        # ==================================
        # 一个简单的 MLP 来提取基本面特征
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # ==================================
        # 3. 融合与预测头 (Head)
        # ==================================
        # 融合后的维度 = LSTM输出(hidden_size) + 表格输出(32)
        fusion_dim = hidden_size + 32
        
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, output_size) # 输出层，直接预测62个值
        )

    def forward(self, x_seq, x_tab):
        """
        Args:
            x_seq: [Batch, 119, 1]  -> 序列数据
            x_tab: [Batch, 27]      -> 表格数据
        """
        
        # --- 1. 处理序列数据 ---
        # LSTM 输出: (output, (h_n, c_n))
        # 我们只需要最后一个时间步的隐状态 h_n，或者 output[:, -1, :]
        # output shape: [Batch, 119, hidden_size]
        lstm_out, _ = self.lstm(x_seq)
        
        # 取最后一个时间步的特征，代表"目前为止的市场状态"
        seq_embedding = lstm_out[:, -1, :] # [Batch, hidden_size]

        # --- 2. 处理静态数据 ---
        tab_embedding = self.tabular_encoder(x_tab) # [Batch, 32]

        # --- 3. 特征融合 ---
        # 将两个特征向量拼接
        combined = torch.cat((seq_embedding, tab_embedding), dim=1) # [Batch, hidden_size + 32]

        # --- 4. 最终预测 ---
        out = self.head(combined) # [Batch, 62]
        
        return out

# ==========================================
# 简单的模型测试代码
# ==========================================
if __name__ == "__main__":
    # 模拟一个 Batch 的数据
    batch_size = 4
    seq_len = 119
    seq_feat = 1
    tab_feat = 27
    
    # 随机生成数据
    dummy_seq = torch.randn(batch_size, seq_len, seq_feat)
    dummy_tab = torch.randn(batch_size, tab_feat)
    
    # 实例化模型
    model = WintonBaselineModel()
    
    # 前向传播
    output = model(dummy_seq, dummy_tab)
    
    print("Input Sequence:", dummy_seq.shape)
    print("Input Tabular:", dummy_tab.shape)
    print("Output Shape:", output.shape) # 应该是 [4, 62]
    print("模型测试通过！可以放入 train.py 中使用了。")