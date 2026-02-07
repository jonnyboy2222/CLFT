import torch
import torch.nn.functional as F
import torch.nn as nn

import math

# torch.erf : 가우시안 형태의 적분을 표현하는 특수함수
def cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gaussian_interval(x, mu, sigma, half_bin=float(0.5), eps=float(1e-12)):

    upper = x + half_bin
    lower = x - half_bin

    z_upper = (upper - mu)/sigma
    z_lower = (lower - mu)/sigma

    p = cdf(z_upper) - cdf(z_lower)

    return p.clamp_min(eps)


class PFIM(nn.Module):
    def __init__(self, in_channels, hidden_channels, eps_sigma=float(1e-4)):
        super().__init__()

        self.eps_sigma = eps_sigma

        self.mu_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, padding=0, bias=True),
        )
        self.sig_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, padding=0, bias=True),
        )


    def forward(self, x):
        mu = self.mu_conv(x)
        sig = self.sig_conv(x)
        # 확률분포 파라미터 양수화 (sigma는 음수가 되면 안되는데 conv출력은 음수도 반환)
        # softplus(z)=log(1+e^z)
        # z가 크면 z, 작으면 0 근처로 부드럽게 (미분이 부드럽게 이어짐)
        sigma = F.softplus(sig) + self.eps_sigma # eps는 0으로 수렴할 때 -log(p)가 무한대로 폭발 방지

        y_hat = torch.round(x)

        p = gaussian_interval(y_hat, mu, sigma)

        nll = -torch.log(p)
        loss = nll.mean()

        info_map = sigma.mean(dim=1, keepdim=True)
        y1 = x * (1 + info_map)

        

        return y1, loss
    

# check
if __name__ == "__main__":
    B, C, H, W = 2, 64, 128, 128
    x = torch.randn(B, C, H, W)
    pfim = PFIM(in_channels=C, hidden_channels=128, kernel_size=1, quant_mode="round")
    y1, loss = pfim(x)
    print(y1.shape, loss.item())