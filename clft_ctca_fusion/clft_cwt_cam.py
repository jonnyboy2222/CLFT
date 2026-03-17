import torch
import torch.nn as nn
import torch.nn.functional as F


class CWT(nn.Module):
    """
    Camera segmentation/CAM guided class prototype tokenizer.

    목적
    ----
    free-slot CWT 대신, 카메라 branch의 class activation prior를 사용해서
    클래스별 semantic prototype token을 직접 만든다.

    핵심
    ----
    1) camera logits/prob map -> class CAM/prior 생성
    2) CAM으로 camera feature를 weighted pooling
    3) pooled feature를 class prototype token으로 사용

    이 방식은 기존 free-slot tokenizer의
    - slot permutation
    - background dominance
    - token collapse
    문제를 구조적으로 줄이는 목적이다.

    입력
    ----
    feat_cam   : (B, D, H, W)
        카메라 branch feature map

    logits_cam : (B, C, H, W)
        카메라 segmentation logits
        보통 background 포함 전체 class logits

    target_class_idx : list[int] 또는 tuple[int], optional
        prototype으로 뽑을 관심 클래스 인덱스.
        예: [1, 2]  # person, car
        None이면 모든 채널 사용.

    반환
    ----
    proto_tokens : (B, K, D)
        클래스별 prototype token

    cam_maps     : (B, K, H, W)
        prototype 추출에 실제 사용된 normalized CAM/prior

    present_mask : (B, K)
        해당 샘플에서 클래스가 충분히 존재하는지 여부

    raw_scores   : (B, K, H, W)
        threshold/topk 전 원본 class score/prob
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        target_class_idx=None,
        use_softmax: bool = True,
        use_sigmoid: bool = False,
        tau: float = 0.0,
        topk_ratio: float = None,
        min_area_ratio: float = 1e-4,
        add_class_queries: bool = True,
        query_init_std: float = 0.02,
        proj: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        if use_softmax and use_sigmoid:
            raise ValueError("Choose only one of use_softmax or use_sigmoid.")
        if not use_softmax and not use_sigmoid:
            raise ValueError("One of use_softmax/use_sigmoid must be True.")

        self.in_dim = in_dim
        self.num_classes = num_classes
        self.target_class_idx = target_class_idx
        self.use_softmax = use_softmax
        self.use_sigmoid = use_sigmoid
        self.tau = tau
        self.topk_ratio = topk_ratio
        self.min_area_ratio = min_area_ratio
        self.add_class_queries = add_class_queries
        self.eps = eps

        if proj:
            self.token_proj = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, in_dim),
            )
        else:
            self.token_proj = nn.Identity()

        # optional residual class anchors for extra stability
        if add_class_queries:
            k = len(target_class_idx) if target_class_idx is not None else num_classes
            self.class_queries = nn.Parameter(torch.randn(k, in_dim) * query_init_std)
        else:
            self.register_parameter("class_queries", None)

    def _select_target_classes(self, logits_cam: torch.Tensor):
        if self.target_class_idx is None:
            return logits_cam
        idx = torch.as_tensor(self.target_class_idx, device=logits_cam.device, dtype=torch.long)
        return logits_cam.index_select(dim=1, index=idx)

    def _build_raw_scores(self, logits_cam: torch.Tensor):
        """
        logits_cam_sel: (B, K, H, W)
        return raw_scores: (B, K, H, W)
        """
        logits_sel = self._select_target_classes(logits_cam)

        if self.use_softmax:
            probs_all = F.softmax(logits_cam, dim=1)
            if self.target_class_idx is None:
                raw_scores = probs_all
            else:
                idx = torch.as_tensor(self.target_class_idx, device=logits_cam.device, dtype=torch.long)
                raw_scores = probs_all.index_select(dim=1, index=idx)
        else:
            raw_scores = torch.sigmoid(logits_sel)

        return raw_scores

    def _apply_threshold_and_topk(self, raw_scores: torch.Tensor):
        """
        raw_scores: (B, K, H, W)
        return refined_scores: (B, K, H, W)
        """
        B, K, H, W = raw_scores.shape
        scores = raw_scores

        if self.tau > 0.0:
            scores = scores * (scores > self.tau).to(scores.dtype)

        if self.topk_ratio is not None:
            if not (0.0 < self.topk_ratio <= 1.0):
                raise ValueError(f"topk_ratio must be in (0,1], got {self.topk_ratio}")

            flat = scores.view(B, K, -1)
            n = flat.shape[-1]
            k_top = max(1, int(round(n * self.topk_ratio)))

            topk_vals, _ = torch.topk(flat, k=k_top, dim=-1)
            thresh = topk_vals[..., -1].unsqueeze(-1)               # (B,K,1)
            keep = (flat >= thresh).to(flat.dtype)
            flat = flat * keep
            scores = flat.view(B, K, H, W)

        return scores

    def _normalize_cam(self, scores: torch.Tensor):
        """
        scores: (B, K, H, W)
        normalize over spatial dimensions so each class map sums to 1.
        """
        denom = scores.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        cam_maps = scores / denom
        return cam_maps

    def _present_mask(self, scores: torch.Tensor):
        B, K, H, W = scores.shape
        area = (scores > 0).to(scores.dtype).sum(dim=(2, 3))   # (B,K)
        min_area = max(1.0, H * W * self.min_area_ratio)
        present = area >= min_area
        return present

    def forward(self, feat_cam: torch.Tensor, logits_cam: torch.Tensor):
        """
        feat_cam   : (B, D, H, W)
        logits_cam : (B, C, H, W)
        """
        if feat_cam.dim() != 4:
            raise ValueError(f"feat_cam must be (B,D,H,W), got {tuple(feat_cam.shape)}")
        if logits_cam.dim() != 4:
            raise ValueError(f"logits_cam must be (B,C,H,W), got {tuple(logits_cam.shape)}")

        B, D, H, W = feat_cam.shape
        if logits_cam.shape[0] != B or logits_cam.shape[2] != H or logits_cam.shape[3] != W:
            raise ValueError(
                f"Spatial mismatch: feat_cam={tuple(feat_cam.shape)}, logits_cam={tuple(logits_cam.shape)}"
            )

        raw_scores = self._build_raw_scores(logits_cam)               # (B,K,H,W)
        refined_scores = self._apply_threshold_and_topk(raw_scores)   # (B,K,H,W)
        present_mask = self._present_mask(refined_scores)             # (B,K)

        # if a class is absent after filtering, keep all-zero map
        refined_scores = refined_scores * present_mask.unsqueeze(-1).unsqueeze(-1).to(refined_scores.dtype)
        cam_maps = self._normalize_cam(refined_scores)                # (B,K,H,W)

        # weighted pooling
        feat_flat = feat_cam.flatten(2)                               # (B,D,HW)
        cam_flat = cam_maps.flatten(2)                                # (B,K,HW)
        proto_tokens = torch.bmm(cam_flat, feat_flat.transpose(1, 2)) # (B,K,D)
        proto_tokens = self.token_proj(proto_tokens)

        if self.add_class_queries:
            proto_tokens = proto_tokens + self.class_queries.unsqueeze(0)

        # zero-out absent classes after adding queries
        proto_tokens = proto_tokens * present_mask.unsqueeze(-1).to(proto_tokens.dtype)

        return proto_tokens, cam_maps, present_mask, raw_scores


class CWTLoss(nn.Module):
    """
    Optional regularization for camera-derived class prototypes.

    구성
    ----
    1) inter-class separation
    2) CAM overlap suppression
    3) running prototype consistency (optional)
    """
    def __init__(
        self,
        num_classes: int,
        lambda_sep: float = 0.05,
        lambda_ov: float = 0.01,
        lambda_proto: float = 0.0,
        momentum: float = 0.9,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_sep = lambda_sep
        self.lambda_ov = lambda_ov
        self.lambda_proto = lambda_proto
        self.momentum = momentum
        self.eps = eps

        if lambda_proto > 0.0:
            self.register_buffer("running_proto", None, persistent=False)
        else:
            self.running_proto = None

    def _sep_loss(self, proto_tokens: torch.Tensor, present_mask: torch.Tensor):
        B, K, D = proto_tokens.shape
        if K <= 1:
            return proto_tokens.new_tensor(0.0)

        x = F.normalize(proto_tokens, dim=-1, eps=self.eps)
        sim = torch.bmm(x, x.transpose(1, 2))  # (B,K,K)

        eye = torch.eye(K, device=sim.device, dtype=torch.bool).unsqueeze(0)
        valid_pair = present_mask.unsqueeze(2) & present_mask.unsqueeze(1) & (~eye)

        if valid_pair.any():
            return sim[valid_pair].mean()
        return proto_tokens.new_tensor(0.0)

    def _overlap_loss(self, cam_maps: torch.Tensor, present_mask: torch.Tensor):
        B, K, H, W = cam_maps.shape
        if K <= 1:
            return cam_maps.new_tensor(0.0)

        overlap_sum = cam_maps.new_tensor(0.0)
        count = 0
        for i in range(K):
            for j in range(i + 1, K):
                valid = present_mask[:, i] & present_mask[:, j]
                if valid.any():
                    ov = (cam_maps[valid, i] * cam_maps[valid, j]).mean()
                    overlap_sum = overlap_sum + ov
                    count += 1
        if count == 0:
            return cam_maps.new_tensor(0.0)
        return overlap_sum / count

    def _proto_loss(self, proto_tokens: torch.Tensor, present_mask: torch.Tensor):
        if self.lambda_proto <= 0.0:
            return proto_tokens.new_tensor(0.0)

        B, K, D = proto_tokens.shape
        x = F.normalize(proto_tokens, dim=-1, eps=self.eps)

        # initialize buffer lazily
        if self.running_proto is None:
            rp = proto_tokens.new_zeros(K, D)
            self.running_proto = rp

        loss = proto_tokens.new_tensor(0.0)
        count = 0
        for k in range(K):
            valid = present_mask[:, k]
            if valid.any():
                cur = x[valid, k].mean(dim=0)
                cur = F.normalize(cur, dim=0, eps=self.eps)
                with torch.no_grad():
                    self.running_proto[k] = self.momentum * self.running_proto[k] + (1 - self.momentum) * cur
                    self.running_proto[k] = F.normalize(self.running_proto[k], dim=0, eps=self.eps)

                rp = self.running_proto[k].detach().unsqueeze(0)
                loss = loss + (1 - F.cosine_similarity(x[valid, k], rp, dim=-1)).mean()
                count += 1

        if count == 0:
            return proto_tokens.new_tensor(0.0)
        return loss / count

    def forward(self, proto_tokens: torch.Tensor, cam_maps: torch.Tensor, present_mask: torch.Tensor):
        loss_sep = self._sep_loss(proto_tokens, present_mask)
        loss_ov = self._overlap_loss(cam_maps, present_mask)
        loss_proto = self._proto_loss(proto_tokens, present_mask)

        total = self.lambda_sep * loss_sep + self.lambda_ov * loss_ov + self.lambda_proto * loss_proto
        logs = {
            "camproto_sep": loss_sep.detach(),
            "camproto_ov": loss_ov.detach(),
            "camproto_proto": loss_proto.detach(),
            "camproto_total": total.detach(),
        }
        return total, logs


