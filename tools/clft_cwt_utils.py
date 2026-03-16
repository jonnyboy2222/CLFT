import torch
import torch.nn.functional as F

def compute_slot_class_score(attn_mean, patch_gt):
    """
    attn_mean: (B,K,Np)
    patch_gt : (B,Np,C)
    return   : (B,K,C)
    """
    return torch.einsum('bkn,bnc->bkc', attn_mean, patch_gt)

def match_slots_k2(slot_class_score):
    """
    slot_class_score: (B,2,2)
    return:
      perm_idx: (B,2)  each row is either [0,1] or [1,0]
    """
    B = slot_class_score.shape[0]
    device = slot_class_score.device

    s00 = slot_class_score[:, 0, 0]
    s11 = slot_class_score[:, 1, 1]
    s01 = slot_class_score[:, 0, 1]
    s10 = slot_class_score[:, 1, 0]

    score_identity = s00 + s11
    score_swap = s01 + s10

    use_swap = score_swap > score_identity

    perm_idx = torch.tensor([[0, 1], [1, 0]], device=device)  # (2,2)
    perm_idx = perm_idx[use_swap.long()]  # (B,2)
    return perm_idx

def reorder_slots(slot_tokens, attn_mean, perm_idx):
    """
    slot_tokens: (B,K,D)
    attn_mean  : (B,K,Np)
    perm_idx   : (B,K)
    """
    B, K, D = slot_tokens.shape
    _, _, Np = attn_mean.shape

    idx_tok = perm_idx.unsqueeze(-1).expand(B, K, D)
    idx_att = perm_idx.unsqueeze(-1).expand(B, K, Np)

    slot_tokens = torch.gather(slot_tokens, dim=1, index=idx_tok)
    attn_mean   = torch.gather(attn_mean, dim=1, index=idx_att)
    return slot_tokens, attn_mean

def slot_assignment_loss(slot_class_score_reordered, class_mask=None, tau=0.2):
    """
    slot_class_score_reordered: (B,K,C), reordered 후
    class_mask: (B,C) optional, 해당 class가 sample에 존재할 때만 loss
    """
    B, K, C = slot_class_score_reordered.shape
    assert K == C

    target = torch.arange(C, device=slot_class_score_reordered.device).unsqueeze(0).expand(B, C)

    # CE 입력: (B*C, C), target: (B*C,)
    logits = (slot_class_score_reordered / tau).reshape(B * K, C)
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

def token_diversity_loss(tok: torch.Tensor):
    # tok: (B, K, D), K=2 가정
    tok = F.normalize(tok, dim=-1)
    cos = F.cosine_similarity(tok[:, 0, :], tok[:, 1, :], dim=-1)  # (B,)
    return (cos ** 2).mean()

def cross_modal_align_loss(tok_cam: torch.Tensor, tok_xyz: torch.Tensor, cls_mask: torch.Tensor):
    # tok_cam/tok_xyz: (B, K, D)
    # cls_mask: (B, K)  1이면 해당 class 존재
    tok_cam = F.normalize(tok_cam, dim=-1)
    tok_xyz = F.normalize(tok_xyz, dim=-1)

    sim = F.cosine_similarity(tok_cam, tok_xyz, dim=-1)  # (B, K)
    loss = 1.0 - sim

    valid = cls_mask.float()
    denom = valid.sum().clamp_min(1.0)
    return (loss * valid).sum() / denom