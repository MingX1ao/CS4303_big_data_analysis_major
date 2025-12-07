import torch
import torch.nn as nn

class WintonBaselineModel(nn.Module):
    """Dual-encoder + dual-head model compatible with your existing pipeline.

    Interface is unchanged: forward(x_seq, x_tab) -> tensor shape [B, 62]

    Design summary:
      - LSTM_intra: models high-frequency minute signals (Ret_2..Ret_120)
      - LSTM_daily: smaller LSTM modeling low-frequency/day-level signals
      - Shared tabular encoder for Feature_* and Ret_MinusTwo/One
      - Two independent heads (intra: 60 outputs, daily: 2 outputs)

    This is the "A: 最小稳定版" architecture discussed earlier.
    """

    def __init__(self,
                 seq_input_size=1,
                 tabular_input_size=27,
                 hidden_size=64,
                 daily_hidden_size=32,
                 output_size=62,
                 dropout_prob=0.3):
        super(WintonBaselineModel, self).__init__()

        # ------------------ intra (minute) encoder ------------------
        # Deeper LSTM for capturing short-term micro-structure
        self.lstm_intra = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout_prob
        )

        # ------------------ daily encoder ------------------
        # Smaller LSTM focused on learning smoother/day-level representations
        self.lstm_daily = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=daily_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout_prob
        )

        # ------------------ tabular encoder (shared) ------------------
        # keep same behaviour as your previous MLP; normalized tabular inputs
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        # ------------------ heads (separate) ------------------
        # intra head -> 60 outputs (minutes)
        intra_fusion_dim = hidden_size + 32
        self.intra_head = nn.Sequential(
            nn.Linear(intra_fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 60)
        )

        # daily head -> 2 outputs (D+1, D+2)
        daily_fusion_dim = daily_hidden_size + 32
        self.daily_head = nn.Sequential(
            nn.Linear(daily_fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 2)
        )

        # small init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.zero_()

    def forward(self, x_seq, x_tab):
        """
        Args:
            x_seq: [B, 119, 1]
            x_tab: [B, 27]
        Returns:
            out: [B, 62] where [:, :60] are minute preds and [:, 60:] are daily preds
        """
        # ---- intra branch ----
        # LSTM returns full sequence; take last time-step
        lstm_out_intra, _ = self.lstm_intra(x_seq)            # [B, 119, hidden_size]
        seq_emb_intra = lstm_out_intra[:, -1, :]             # [B, hidden_size]

        # ---- daily branch ----
        lstm_out_daily, _ = self.lstm_daily(x_seq)            # [B, 119, daily_hidden_size]
        seq_emb_daily = lstm_out_daily[:, -1, :]             # [B, daily_hidden_size]

        # ---- tabular encoding (shared) ----
        # BatchNorm1d expects [B, C]
        tab_emb = self.tabular_encoder(x_tab)                # [B, 32]

        # ---- fusion & heads ----
        intra_comb = torch.cat((seq_emb_intra, tab_emb), dim=1)   # [B, hidden+32]
        daily_comb = torch.cat((seq_emb_daily, tab_emb), dim=1)   # [B, daily_hidden+32]

        out_intra = self.intra_head(intra_comb)   # [B, 60]
        out_daily = self.daily_head(daily_comb)   # [B, 2]

        out = torch.cat((out_intra, out_daily), dim=1)   # [B, 62]
        return out


# ============================
# Quick smoke test (keeps same interface as before)
# ============================
if __name__ == "__main__":
    batch_size = 4
    seq_len = 119
    seq_feat = 1
    tab_feat = 27

    dummy_seq = torch.randn(batch_size, seq_len, seq_feat)
    dummy_tab = torch.randn(batch_size, tab_feat)

    model = WintonBaselineModel()
    out = model(dummy_seq, dummy_tab)
    print("Output Shape:", out.shape)  # should be [4, 62]
