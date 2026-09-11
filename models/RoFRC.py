import sys
import torch
from toolkit.framework import FewShotREModel

sys.path.append('..')


class RoFRC(FewShotREModel):
    """
    RoFRC adapted for BaFRC's training/evaluation interface.

    Original:
      Hu et al. 2025 - "Towards Robust Open-set Few-shot Relation Classification"

    Interface changes (original → adapted):
      forward()  4 return values → 3: (logits, pred, None)
      loss()     (logits, rel, proto, label, N, Q)
               → (logits, logits_ova, label, N, Q)
                  logits_ova is ignored; agreement loss disabled

    Model logic (unchanged):
      support_mean = mean_K(support_embeddings)          (B, N, 2D)
      rel_text_emb = CLS_emb cat avg_token_emb           (B, N+1, 2D)
      logits_proto    = dot(support_mean, query)         (B, total_Q, N)
      logits_relation = dot(rel_text_emb, query)         (B, total_Q, N+1)
      logits_known    = (logits_proto + logits_rel[:N]) / 2
      logits          = [logits_known | logits_rel[:,N]] (B, total_Q, N+1)
      pred            = argmax(logits)   ∈ {0,...,N},  N = NOTA label

    Data requirement:
      rel_text must have N+1 entries per episode (N known classes + 1 NOTA).
      Use BaFRC's FewRelDataset with use_std_desc=False.
      pid2name format: {pid: [name, description]}  (2-element list is fine;
      BaFRC's dataset checks len(entry) > 1 before reading orig_desc).
    """

    def __init__(self, sentence_encoder, hidden_size, max_len, lamb=1e-5):
        FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.max_len = max_len
        # lamb kept for API compatibility; agreement loss is currently disabled
        self.register_buffer("lamb", torch.tensor(float(lamb)))

    @staticmethod
    def _dot(x, y, dim):
        return (x * y).sum(dim)

    @staticmethod
    def _batch_dot(S, Q):
        """
        S: (B, n, d)  Q: (B, m, d)  →  (B, m, n)
        Uses unsqueeze to broadcast:
          S.unsqueeze(1): (B, 1, n, d)
          Q.unsqueeze(2): (B, m, 1, d)
          product sum over last dim: (B, m, n)
        """
        return (S.unsqueeze(1) * Q.unsqueeze(2)).sum(-1)

    def forward(self, support, query, rel_text, N, K, Q, total_Q):
        """
        support:  dict, instances, (B*N*K, ...)
        query:    dict, instances, (B*total_Q, ...)
        rel_text: dict, relation descriptions, (B*(N+1), ...)
                  indices 0..N-1: known relations, index N: NOTA

        Returns:
          logits: (B, total_Q, N+1)
          pred:   (B, total_Q)  values in [0, N], N = NOTA
          None:   placeholder (logits_ova not used)
        """
        support_glo, _  = self.sentence_encoder(support)               # (B*N*K, 2D)
        query_glo,   _  = self.sentence_encoder(query)                  # (B*total_Q, 2D)
        rel_glo, rel_loc = self.sentence_encoder(rel_text, cat=False)  # (B*(N+1), D) each

        B = support_glo.size(0) // (N * K)
        D = self.hidden_size * 2

        support_emb = support_glo.view(B, N, K, D)                      # (B, N, K, 2D)
        query_emb   = query_glo.view(B, total_Q, D)                     # (B, total_Q, 2D)

        # rel_text encoding: CLS ⊕ avg(token embeddings)
        rel_avg = rel_loc.mean(dim=1)                                    # (B*(N+1), D)
        rel_emb = torch.cat([rel_glo, rel_avg], dim=-1)                 # (B*(N+1), 2D)
        rel_emb = rel_emb.view(B, N + 1, D)                             # (B, N+1, 2D)

        support_mean = support_emb.mean(dim=2)                          # (B, N, 2D)

        logits_proto    = self._batch_dot(support_mean, query_emb)      # (B, total_Q, N)
        logits_relation = self._batch_dot(rel_emb, query_emb)           # (B, total_Q, N+1)

        # NOTA logit comes exclusively from relation text (no support prototype for NOTA)
        logits_nota  = logits_relation[:, :, N]                         # (B, total_Q)
        logits_known = (logits_proto + logits_relation[:, :, :N]) / 2  # (B, total_Q, N)

        logits = torch.cat(
            [logits_known, logits_nota.unsqueeze(-1)], dim=-1           # (B, total_Q, N+1)
        )

        pred = logits.view(B * total_Q, N + 1).argmax(dim=-1).view(B, total_Q)

        return logits, pred, None

    def loss(self, logits, logits_ova, query_label, N, Q):
        """
        logits:      (B, total_Q, N+1)
        logits_ova:  None (BaFRC framework always passes None here)
        query_label: (B, total_Q), values in [0..N], N = NOTA

        Returns CE loss over N+1 classes.
        Agreement loss (original paper) is kept disabled as in the released code.
        """
        B, total_Q, n_cls = logits.shape
        return self.cost(logits.view(-1, n_cls), query_label.view(-1))
