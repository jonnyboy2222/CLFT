from clft import CLFT
import torch
import json

with open("config.json", "r") as f:
    config = json.load(f)


resize = config['Dataset']['transforms']['resize']
model = CLFT(RGB_tensor_size=(3, resize, resize),
                              XYZ_tensor_size=(3, resize, resize),
                              patch_size=config['CLFT']['patch_size'],
                              emb_dim=config['CLFT']['emb_dim'],
                              resample_dim=config['CLFT']['resample_dim'],
                              read=config['CLFT']['read'],
                              hooks=config['CLFT']['hooks'],
                              reassemble_s=config['CLFT']['reassembles'],
                              nclasses=len(config['Dataset']['classes']),
                              type=config['CLFT']['type'],
                              model_timm=config['CLFT']['model_timm'],)
ckpt = torch.load("/home/john/dev_ws/CLFT/logs/clft_aux_ups_v2_fusioncheckpoint_90.pth", map_location="cpu")

# swin용
# dummy forward로 conv1 실제 생성
# device = torch.device("cuda")
# modality = "cross_fusion"
# model.to(device)
# model.eval()
# with torch.no_grad():
#     dummy_rgb = torch.zeros(1, 3, resize, resize, device=device)
#     dummy_lidar = torch.zeros(1, 3, resize, resize, device=device)
#     _ = model(dummy_rgb, dummy_lidar, modality)  # 여기서 reassemble.resample.conv1들이 다 만들어짐
# model.load_state_dict(ckpt["model_state_dict"])

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params: {total:,}")
print(f"Trainable params: {trainable:,}")
print(f"Params (M): {total/1e6:.2f}M")
