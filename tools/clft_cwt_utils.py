import torch

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