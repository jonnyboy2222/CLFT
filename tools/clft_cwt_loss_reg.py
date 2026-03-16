import torch
import torch.nn.functional as F

def slot_assignment_loss(slot_class_score_reordered, class_mask=None):
    """
    slot_class_score_reordered: (B,K,C), reordered 후
    class_mask: (B,C) optional, 해당 class가 sample에 존재할 때만 loss
    """
    B, K, C = slot_class_score_reordered.shape
    assert K == C

    target = torch.arange(C, device=slot_class_score_reordered.device).unsqueeze(0).expand(B, C)

    # CE 입력: (B*C, C), target: (B*C,)
    logits = slot_class_score_reordered.reshape(B * K, C)
    tgt = target.reshape(B * K)

    loss_all = F.cross_entropy(logits, tgt, reduction='none').view(B, K)

    if class_mask is not None:
        # slot k는 class k에 대응하므로 mask도 같은 인덱스를 사용
        valid = class_mask.float()  # (B,C)
        loss_all = loss_all * valid
        denom = valid.sum().clamp_min(1.0)
        return loss_all.sum() / denom

    return loss_all.mean()

def attention_diversity_loss(attn_mean):
    """
    attn_mean: (B,K,Np)
    """
    B, K, Np = attn_mean.shape

    # normalize on patch dimension
    A = F.normalize(attn_mean, p=2, dim=-1)   # (B,K,Np)
    G = torch.bmm(A, A.transpose(1, 2))       # (B,K,K)

    I = torch.eye(K, device=attn_mean.device).unsqueeze(0)  # (1,K,K)
    loss = ((G - I) ** 2).mean()
    return loss