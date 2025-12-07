import torch
import torch.nn as nn

class WintonBaselineModel(nn.Module):
    """Dual-encoder + dual-head model with intraday-to-daily aggregation.

    Interface unchanged: forward(x_seq, x_tab) -> [B, 62]
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
        self.lstm_intra = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout_prob
        )

        # ------------------ daily encoder ------------------
        # +5 from aggregated intraday features
        self.daily_encoder = nn.Sequential(
            nn.Linear(tabular_input_size + 5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, daily_hidden_size),
            nn.BatchNorm1d(daily_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
        )

        # ------------------ tabular encoder (shared) ------------------
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        # ------------------ heads ------------------
        intra_fusion_dim = hidden_size + 32
        self.intra_head = nn.Sequential(
            nn.Linear(intra_fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 60)
        )

        daily_fusion_dim = daily_hidden_size + 32
        self.daily_head = nn.Sequential(
            nn.Linear(daily_fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 2)
        )

        self._init_weights()

    # ------------------ intraday → daily aggregation ------------------
    def _aggregate_intraday(self, x_seq):
        """
        x_seq: [B, 119, 1]
        returns: [B, 5]
        """
        x = x_seq.squeeze(-1)        # [B, 119]

        sum_all = x.sum(dim=1, keepdim=True)
        sum_last30 = x[:, -30:].sum(dim=1, keepdim=True)

        std_all = x.std(dim=1, keepdim=True)

        max_abs = x.abs().max(dim=1, keepdim=True)[0]

        # linear trend slope
        t = torch.arange(x.shape[1], device=x.device).float()  # [119]
        t = t - t.mean()
        slope = (x * t).sum(dim=1, keepdim=True) / (t.pow(2).sum() + 1e-6)

        daily_agg = torch.cat(
            [sum_all, sum_last30, std_all, max_abs, slope], dim=1
        )  # [B, 5]

        return daily_agg

    # ------------------ weight init ------------------
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

    # ------------------ forward ------------------
    def forward(self, x_seq, x_tab):
        """
        Args:
            x_seq: [B, 119, 1]
            x_tab: [B, 27]
        Returns:
            out: [B, 62]
        """

        # ---- intra branch ----
        lstm_out_intra, _ = self.lstm_intra(x_seq)
        seq_emb_intra = lstm_out_intra[:, -1, :]

        # ---- intraday → daily aggregation ----
        daily_agg = self._aggregate_intraday(x_seq)   # [B, 5]
        daily_input = torch.cat([x_tab, daily_agg], dim=1)  # [B, 32]

        # ---- daily branch ----
        seq_emb_daily = self.daily_encoder(daily_input)

        # ---- tabular encoding (shared) ----
        tab_emb = self.tabular_encoder(x_tab)

        # ---- fusion & heads ----
        intra_comb = torch.cat((seq_emb_intra, tab_emb), dim=1)
        daily_comb = torch.cat((seq_emb_daily, tab_emb), dim=1)

        out_intra = self.intra_head(intra_comb)
        out_daily = self.daily_head(daily_comb)

        out = torch.cat((out_intra, out_daily), dim=1)
        return out


# ============================
# Quick smoke test
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
    print("Output Shape:", out.shape)  # [4, 62]
