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
    def __init__(self, in_channels, hidden_channels, eps_sigma=float(0.01)):
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


    def forward(self, x, gain_gate=None):
        # mu = self.mu_conv(x)
        mu = x.detach()
        sig = self.sig_conv(x)
        # 확률분포 파라미터 양수화 (sigma는 음수가 되면 안되는데 conv출력은 음수도 반환)
        # softplus(z)=log(1+e^z)
        # z가 크면 z, 작으면 0 근처로 부드럽게 (미분이 부드럽게 이어짐)
        sigma = F.softplus(sig) + self.eps_sigma # eps는 0으로 수렴할 때 -log(p)가 무한대로 폭발 방지

        # quantized observation
        # train: additive uniform noise for differentiable quantization proxy
        # eval : hard rounding
        if self.training:
            y_hat = x + (torch.rand_like(x) - 0.5)
        else:
            y_hat = torch.round(x)

        p = gaussian_interval(y_hat, mu, sigma)

        nll = -torch.log2(p)                       # (B,C,H,W)
        loss = nll.mean()

        # if weight_map is None:
        #     loss = nll.mean()
        # else:
        #     # weight_map: (B,1,H,W) or (B,H,W)
        #     if weight_map.ndim == 3:
        #         weight_map = weight_map.unsqueeze(1)
        #     w = weight_map.to(dtype=nll.dtype)    # (B,1,H,W)
        #     w = w.expand_as(nll)                  # (B,C,H,W)

        #     loss = (nll * w).sum() / (w.sum() + 1e-12)

        info_map = sigma.mean(dim=1, keepdim=True)
        # y1 = x * (1 + info_map)

        # gain gate추가 (증폭을 객체쪽으로 하도록 수행)
        if gain_gate is not None:
            if gain_gate.ndim == 3:
                gain_gate = gain_gate.unsqueeze(1)
            gain_gate = gain_gate.to(dtype=info_map.dtype, device=info_map.device)

            if gain_gate.shape[-2:] != info_map.shape[-2:]:
                gain_gate = F.interpolate(
                    gain_gate,
                    size=info_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            gain_gate = gain_gate.clamp(0.0, 1.0)
            gain = 1.0 + info_map * gain_gate
        else:
            gain = 1.0 + info_map

        y1 = x * gain

        with torch.no_grad():
            info = info_map
            info_flat = info.flatten(1)
            p_flat = p.flatten(1)
            gain = 1.0 + info

            self.diag = {
                "pfim_loss": float(loss.detach().cpu()),

                # info_map / sigma stats
                "info_mean": float(info.mean().detach().cpu()),
                "info_std": float(info.std().detach().cpu()),
                "info_min": float(info.min().detach().cpu()),
                "info_p05": float(info_flat.quantile(0.05, dim=1).mean().detach().cpu()),
                "info_p50": float(info_flat.quantile(0.50, dim=1).mean().detach().cpu()),
                "info_p95": float(info_flat.quantile(0.95, dim=1).mean().detach().cpu()),
                "info_max": float(info.max().detach().cpu()),
                "sigma_min": float(sigma.min().detach().cpu()),
                "sigma_p95": float(sigma.flatten(1).quantile(0.95, dim=1).mean().detach().cpu()),

                # p stats
                "p_mean": float(p.mean().detach().cpu()),
                "p_p05": float(p_flat.quantile(0.05, dim=1).mean().detach().cpu()),
                "p_p95": float(p_flat.quantile(0.95, dim=1).mean().detach().cpu()),
                "p_clamp_ratio": float((p <= 1e-12 * 1.01).float().mean().detach().cpu()),

                # mu-fit / z
                "mae_mu_x": float((mu - x).abs().mean().detach().cpu()),
                "mae_yhat_mu": float((y_hat - mu).abs().mean().detach().cpu()),
                "z_abs_mean": float(((y_hat - mu).abs() / sigma).mean().detach().cpu()),

                # actual amplification
                "gain_mean": float(gain.mean().detach().cpu()),
                "gain_p95": float(gain.flatten(1).quantile(0.95, dim=1).mean().detach().cpu()),
            }

        # viz
        with torch.no_grad():
            # info_map: (B, 1, H, W) or (B, C, H, W) -> 시각화는 보통 (B,1,H,W) 사용
            self.last_info_map = info_map.detach()

            # gain = 1 + info_map
            self.last_gain_map = (1.0 + info_map).detach()


        

        return y1, loss, info_map
    

# check
# if __name__ == "__main__":
#     B, C, H, W = 2, 64, 128, 128
#     x = torch.randn(B, C, H, W)
#     pfim = PFIM(in_channels=C, hidden_channels=128, kernel_size=1, quant_mode="round")
#     y1, loss = pfim(x)
#     print(y1.shape, loss.item())