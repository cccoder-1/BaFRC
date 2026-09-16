import sys
from contextlib import nullcontext
from toolkit.framework import FewShotREModel
from typing import Optional
import torch
import torch.nn.functional as F

sys.path.append('..')


class BaFRC(FewShotREModel):
    """
    BaFRC model with description-enhanced prototypes and boundary-aware matching.

    For each class c:
      - support embeddings: s_{c,i}, i=1..K
      - original description embedding: r_c^orig
      - standardized description embedding: r_c^std

    Prototype:
      - use_std_desc=True:  p_c = (sum_i s_{c,i} + r_c^orig + r_c^std) / (K + 2)
      - use_std_desc=False: p_c = (sum_i s_{c,i} + r_c^orig) / (K + 1)

    Classification (main loss):
      - N-way cross entropy on known queries only (label < N)
      - NOTA is decided at inference by nearest prototype + radius reject.

    Boundary loss (paper Eq. 13-15):
      X_c^+ = { known queries with label == c }
      X_c^- = { queries with label != c }  (includes NOTA queries, label == N)
      R_c   = h_v({ d(p_c, x^-) - m })  (quantile over negatives)

      ℓ_c^+ = (1/γ) log(1 + Σ exp( γ (d(x,p_c) − R_c)    ))   x∈X_c^+
      ℓ_c^- = (1/γ) log(1 + Σ exp(-γ (d(x,p_c) − R̃_c − m)))   x∈X_c^-
      L_B   = 1/N Σ_c [ ℓ_c^+ + ℓ_c^- ]
      The two terms can be disabled independently with bafrc_loss_pos and
      bafrc_loss_neg for ablation studies.

    Total (paper Eq. 16):
      L = L_CE + L_B
    """

    def __init__(
        self,
        sentence_encoder,
        hidden_size,
        max_len,
        cls_temp=10.0,
        bafrc_gamma=3.0,
        bafrc_margin=0.15,
        radius_quantile=0.1,
        radius_reg=0.1,  # legacy compatibility only; no longer used in the loss
        dist_type="euclidean",
        # legacy/unused args kept for compatibility
        use_query_in_radius=True,
        use_desc_neg=True,
        use_query_in_boundary=False,
        radius_update_interval=100,
        use_std_desc=True,
        radius_blend_rho=0.5,
        radius_detach=True,
        radius_max=10.0,
        # Active L_B switches; the radius regularization switch is legacy-only.
        bafrc_loss_pos=True,
        bafrc_loss_neg=True,
        bafrc_loss_radius_reg=True,  # legacy compatibility only
        # Legacy MRM keyword aliases. Prefer bafrc_* in new code.
        mrm_gamma=None,
        mrm_margin=None,
        mrm_loss_pos=None,
        mrm_loss_neg=None,
        mrm_loss_radius_reg=None,
    ):
        FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.use_std_desc = bool(use_std_desc)

        if mrm_gamma is not None:
            bafrc_gamma = mrm_gamma
        if mrm_margin is not None:
            bafrc_margin = mrm_margin
        if mrm_loss_pos is not None:
            bafrc_loss_pos = mrm_loss_pos
        if mrm_loss_neg is not None:
            bafrc_loss_neg = mrm_loss_neg
        if mrm_loss_radius_reg is not None:
            bafrc_loss_radius_reg = mrm_loss_radius_reg

        self.cls_temp = float(cls_temp)
        self.gamma = float(bafrc_gamma)
        self.margin = float(bafrc_margin)
        self.v = float(radius_quantile)
        self.lam_r = float(radius_reg)  # retained only for loading/running legacy commands
        self.radius_blend_rho = float(radius_blend_rho)
        self.radius_detach = bool(radius_detach)
        self.radius_max = None if radius_max is None or float(radius_max) <= 0 else float(radius_max)

        self.bafrc_loss_pos = bool(bafrc_loss_pos)
        self.bafrc_loss_neg = bool(bafrc_loss_neg)
        self.bafrc_loss_radius_reg = bool(bafrc_loss_radius_reg)

        self.dist_type = str(dist_type).lower()
        if self.dist_type not in {"euclidean", "cosine"}:
            raise ValueError(f"Unsupported dist_type={dist_type}. Use 'euclidean' or 'cosine'.")

        self.eps = 1e-8
        self._cache = {}

    def _cosine_distance(self, Q, P):
        sim = torch.matmul(Q, P.transpose(1, 2))
        return 1.0 - sim

    def _euclidean_distance(self, Q, P):
        return torch.norm(Q.unsqueeze(2) - P.unsqueeze(1), p=2, dim=-1)

    def _distance(self, Q, P):
        if self.dist_type == "cosine":
            Qn = F.normalize(Q, p=2, dim=-1)
            Pn = F.normalize(P, p=2, dim=-1)
            return self._cosine_distance(Qn, Pn)
        return self._euclidean_distance(Q, P)

    def _encode_rel_pair_no_nota(self, rel_text, B: int, N: int):
        """
        use_std_desc=True: the first 2N entries are [c0_orig,c0_std,...].
        use_std_desc=False: the first N entries are class originals.
        Any trailing NOTA entries are excluded from this slice.
        """
        rel_text_glo, rel_text_loc = self.sentence_encoder(rel_text, cat=False)  # (B*M, D), (B*M, L, D)
        rel_text_loc = torch.mean(rel_text_loc, dim=1)
        rel_h = torch.cat((rel_text_glo, rel_text_loc), dim=-1)  # (B*M, 2D)

        M = rel_h.size(0) // B
        rel_h = rel_h.view(B, M, self.hidden_size * 2)
        if self.use_std_desc:
            if M < 2 * N:
                raise RuntimeError(f"rel_text has too few entries: got M={M}, expected >= {2*N}")
            rel_known = rel_h[:, : 2 * N, :]
            rel_orig = rel_known[:, 0::2, :]
            rel_std = rel_known[:, 1::2, :]
        else:
            if M < N:
                raise RuntimeError(f"rel_text has too few entries: got M={M}, expected >= {N}")
            rel_orig = rel_h[:, :N, :]
            rel_std = rel_orig.new_zeros(B, N, rel_orig.size(-1))
        return rel_orig, rel_std

    def _build_prototypes(self, support_h: torch.Tensor, rel_orig: torch.Tensor, rel_std: torch.Tensor) -> torch.Tensor:
        """
        support_h: (B, N, K, d)
        rel_orig/std: (B, N, d); without std, rel_std is an unused placeholder.
        """
        B, N, K, d = support_h.shape
        sup_sum = support_h.sum(dim=2)  # (B,N,d)
        k_eff = support_h.new_full((B, N, 1), float(K))
        if self.use_std_desc:
            P = (sup_sum + rel_orig + rel_std) / (k_eff + 2.0)
        else:
            P = (sup_sum + rel_orig) / (k_eff + 1.0)
        return P

    def _radius_from_support(self, support_h: torch.Tensor, P: torch.Tensor, N: int) -> torch.Tensor:
        """
        support_h: (B,N,K,d), P: (B,N,d)
        R_sup from other-class supports.
        """
        B, N0, K, d = support_h.shape
        assert N0 == N
        support_flat = support_h.reshape(B, N * K, d)
        dists_support = self._distance(support_flat, P)  # (B, N*K, N)
        support_cls = torch.arange(N, device=support_h.device).repeat_interleave(K)  # (N*K,)
        v = min(max(float(self.v), 0.0), 1.0)
        R_max = 2.0 if self.dist_type == "cosine" else self.radius_max

        R = support_h.new_zeros(B, N)
        for b in range(B):
            for c in range(N):
                neg_mask = (support_cls != c)
                neg_vals = dists_support[b, neg_mask, c]
                shifted = neg_vals - float(self.margin)
                if shifted.numel() == 0:
                    R[b, c] = float(self.eps)
                else:
                    rbc = torch.quantile(shifted, q=v)
                    R[b, c] = torch.clamp(rbc, min=self.eps, max=R_max) if R_max is not None else torch.clamp(rbc, min=self.eps)
        return R

    def _radius_from_neg_queries(self, dists_q: torch.Tensor, query_label: torch.Tensor, N: int) -> torch.Tensor:
        """
        R_c = quantile({ d(q, p_c) - margin | label(q) != c }, v)
        """
        B = dists_q.size(0)
        device = dists_q.device
        v = min(max(float(self.v), 0.0), 1.0)
        R_max = 2.0 if self.dist_type == "cosine" else self.radius_max

        R = torch.full((B, N), self.eps, device=device, dtype=dists_q.dtype)
        for b in range(B):
            for c in range(N):
                neg_vals = dists_q[b, :, c][query_label[b] != c]
                if neg_vals.numel() == 0:
                    continue
                shifted = neg_vals - float(self.margin)
                rbc = torch.quantile(shifted, q=v)
                R[b, c] = torch.clamp(rbc, min=self.eps, max=R_max) if R_max is not None else torch.clamp(rbc, min=self.eps)
        return R

    # ------------------------------------------------------------------
    # Inference-time Boundary Calibration helpers
    # ------------------------------------------------------------------

    def compute_support_radius_base(self, support_h: torch.Tensor, P: torch.Tensor, N: int) -> torch.Tensor:
        """
        Returns the raw support-based quantile radius BEFORE subtracting any margin.

        R_base[b, c] = quantile_v({ d(s, p_c) | s belongs to other classes })

        Relationship to _radius_from_support:
            _radius_from_support  →  clamp(R_base - self.margin)
            predict_with_eval_margin(eval_margin)  →  clamp(R_base - eval_margin)

        Therefore predict_with_eval_margin(..., eval_margin=self.margin) exactly
        reproduces the standard forward() rejection boundary.
        This allows sweeping eval_margin without rerunning the BERT encoder.
        """
        B, N0, K, d = support_h.shape
        assert N0 == N
        support_flat = support_h.reshape(B, N * K, d)
        dists_support = self._distance(support_flat, P)          # (B, N*K, N)
        support_cls = torch.arange(N, device=support_h.device).repeat_interleave(K)
        v = min(max(float(self.v), 0.0), 1.0)

        R_base = support_h.new_zeros(B, N)
        for b in range(B):
            for c in range(N):
                neg_mask = (support_cls != c)
                neg_vals = dists_support[b, neg_mask, c]
                if neg_vals.numel() == 0:
                    R_base[b, c] = 0.0
                else:
                    R_base[b, c] = torch.quantile(neg_vals, q=v)
        return R_base                                             # (B, N), no clamp yet

    def predict_with_eval_margin(
        self,
        dists_q: torch.Tensor,
        radius_base: torch.Tensor,
        eval_margin: float,
        N: int,
    ) -> torch.Tensor:
        """
        Produce predictions using a specified inference-time margin.

        R(eval_margin) = clamp(R_base - eval_margin, eps, R_max)
        pred = argmin_c d(q, p_c)
        pred[min_d > R_nearest] = N   (NOTA)

        Does NOT modify self.margin or any model parameters.
        eval_margin == self.margin reproduces the standard forward() result.

        Args:
            dists_q:     (B, total_Q, N)  query-to-prototype distances
            radius_base: (B, N)           from compute_support_radius_base()
            eval_margin: float            inference-time margin to subtract
            N:           int              number of known classes
        Returns:
            pred: (B, total_Q) with values in {0..N-1, N} where N = NOTA
        """
        R_max = 2.0 if self.dist_type == "cosine" else self.radius_max
        raw_radius = radius_base - eval_margin
        R = torch.clamp(raw_radius, min=self.eps, max=R_max) if R_max is not None else torch.clamp(raw_radius, min=self.eps)
        min_d, nearest = dists_q.min(dim=-1)                                   # (B, total_Q)
        R_nearest = R.gather(1, nearest)                                        # (B, total_Q)
        pred = nearest.clone()
        pred[min_d > R_nearest] = N
        return pred

    def forward(self, support, query, rel_text, N, K, Q, total_Q):
        support_h, _ = self.sentence_encoder(support)  # (B*N*K, d)
        query_h, _ = self.sentence_encoder(query)      # (B*total_Q, d)

        BNK = support_h.size(0)
        B = max(1, BNK // (N * K))

        support_h = support_h.view(B, N, K, self.hidden_size * 2)
        query_h = query_h.view(B, total_Q, self.hidden_size * 2)

        rel_orig, rel_std = self._encode_rel_pair_no_nota(rel_text, B, N)
        P = self._build_prototypes(support_h, rel_orig, rel_std)  # (B,N,d)

        dists_q = self._distance(query_h, P)  # (B,total_Q,N)
        logits_nway = -self.cls_temp * dists_q

        # Inference-time radius from supports (same support set used to build prototypes).
        radius_context = torch.no_grad() if self.radius_detach else nullcontext()
        with radius_context:
            R = self._radius_from_support(support_h, P, N)  # (B,N)

        # cache for training loss
        self._cache = {
            "dists_q": dists_q,   # query-to-prototype distances
            "R_support": R,       # support-calibrated radius (also used by inference)
            "support_h": support_h,
            "P": P,
        }

        min_d, nearest = dists_q.min(dim=-1)  # (B,total_Q)
        R_nearest = R.gather(1, nearest)
        pred = nearest.clone()
        pred[min_d > R_nearest] = N

        dummy = logits_nway.new_zeros(B, total_Q, N)
        return logits_nway, pred, dummy

    def loss(self, logits_nway, logits_ova, query_label, N, Q):
        """
        logits_nway: (B,total_Q,N)
        query_label: (B,total_Q) with NOTA label == N
        """
        B, T, n_cls = logits_nway.shape
        assert n_cls == N

        # main CE on known only
        known = (query_label < N)
        if known.any():
            loss_ce = F.cross_entropy(logits_nway[known], query_label[known])
        else:
            # Keep a zero loss connected to the graph for pure-NOTA batches.
            loss_ce = logits_nway.sum() * 0.0

        dists_q = self._cache.get("dists_q", None)
        if dists_q is None:
            raise RuntimeError("BaFRC.loss called before forward (missing cached distances).")

        # Training radius: support-calibrated + query-refined
        #   R_train = (1-rho) * R_sup + rho * R_qry
        #   rho=1.0 -> pure query-based (FewRel paper setting)
        #   rho=0.5 -> blended (FS-TACRED appendix setting)
        #   rho=0.0 -> pure support (train/inference-consistent ablation)
        radius_context = torch.no_grad() if self.radius_detach else nullcontext()
        with radius_context:
            R_sup = self._cache.get("R_support", None)
            if R_sup is None:
                raise RuntimeError("BaFRC.loss called before forward (missing cached support radius).")
            R_qry = self._radius_from_neg_queries(dists_q, query_label, N)  # (B,N)
            rho = min(max(float(self.radius_blend_rho), 0.0), 1.0)
            R = (1.0 - rho) * R_sup + rho * R_qry

        gamma = max(self.gamma, self.eps)
        m = float(self.margin)

        loss_b = logits_nway.new_tensor(0.0)
        for c in range(N):
            per_batch = logits_nway.new_tensor(0.0)
            for b in range(B):
                R_bc = R[b, c]

                # ℓ_c^+: positive penalty (Eq. 13)
                d_pos = dists_q[b, :, c][query_label[b] == c]
                if d_pos.numel() > 0:
                    z_pos = gamma * (d_pos - R_bc)
                    pos_term_b = torch.log1p(torch.exp(z_pos).sum()) / gamma
                else:
                    pos_term_b = logits_nway.new_tensor(0.0)

                # ℓ_c^-: negative penalty (Eq. 14), X_c^- includes NOTA
                d_neg = dists_q[b, :, c][query_label[b] != c]
                if d_neg.numel() > 0:
                    z_neg = -gamma * (d_neg - (R_bc + m))
                    neg_term_b = torch.log1p(torch.exp(z_neg).sum()) / gamma
                else:
                    neg_term_b = logits_nway.new_tensor(0.0)

                if self.bafrc_loss_pos:
                    per_batch = per_batch + pos_term_b
                if self.bafrc_loss_neg:
                    per_batch = per_batch + neg_term_b

            loss_b = loss_b + per_batch / float(B)

        loss_b = loss_b / float(N)
        # L = L_CE + L_B  (Eq. 16, no separate weight)
        return loss_ce + loss_b
