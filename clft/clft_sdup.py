import torch
import torch.nn as nn

class SDUP(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1)
        )

    def forward(self, x):
        u_logit = self.head(x)
        u_map = torch.sigmoid(u_logit)
        y1 = x * (1.0 + u_map)

        with torch.no_grad():
            self.last_info_map = u_map.detach()
            self.last_gain_map = (1.0 + u_map).detach()

        return y1, u_map