import torch.nn as nn
import torch
import torch.nn.functional as F

def gaussian(H, W, cx, cy):
    sigma = [H**2, 0, 
             0, W**2]
    p = (1/torch.sqrt(2 * torch.pi) * sigma**2) * torch.e**()

class PGDPBuilderGT:
    '''
    M_gt 로직
        - gt를 기반으로 
        - 한 채널로 합친 gaussian distribution map에서
        - 전 픽셀 값 평균내고 (threshold)
        - 임계값 넘는 값이면 (논문에서는 sign함수) 1, 아니면 -1
        -> 공식 : (sign + 1) * 0.25 + p (각 픽셀의 gaussian 값)
        (클래스 밖에서 M_gt와 MSE weighted loss - 10 fg, 0.1 bg)
    '''
    def __init__(self):
        super().__init__()



    def __call__(self, H, W, cx, cy):
        # annotation을 받아서 bbox처리는 밖에서
        # mask -> bbox?

        p = gaussian(H, W, cx, cy)
        th = p.mean(dim=(2,3), keepdim=True)
        sign = torch.where(p >= th, 1.0, -1.0)
        
        M_gt = (sign + 1) * 0.25 + p

        return M_gt



