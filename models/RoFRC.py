import sys
from toolkit.framework import FewShotREModel
import torch

sys.path.append('..')


class RoFRC(FewShotREModel):

    def __init__(self, sentence_encoder, hidden_size, max_len, lamb=1e-5):
        FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.max_len = max_len
        if torch.cuda.is_available():
            self.lamb = torch.tensor(float(lamb), requires_grad=False).cuda()
        else:
            self.lamb = torch.tensor(float(lamb), requires_grad=False)

    def __dist__(self, x, y, dim):
        return (x * y).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def loss(self, logits, logits_relation_for_loss, logits_proto_for_loss, query_label, N, Q):
        """
        :param logits: (B, total_Q, N + 1)
        :param query_label: (B, total_Q)
        :return:
        """
        B = logits.shape[0]
        total_Q = logits.shape[1]
        loss_softmax = self.cost(logits.view(-1, N + 1), query_label.view(-1))

        logits_proto_for_loss = torch.softmax(logits_proto_for_loss, dim=-1)
        logits_relation_for_loss = torch.softmax(logits_relation_for_loss, dim=-1)
        p_1_1, idx = logits_proto_for_loss.max(dim=-1)
        p_1_0 = 1. - p_1_1
        p_2_1 = torch.gather(logits_relation_for_loss, dim=-1, index=idx.view(B, total_Q, 1))
        p_2_1 = p_2_1.view(B, total_Q)
        p_2_0 = 1. - p_2_1

        adv_loss_non = (- torch.log(p_1_1 * p_2_0 + p_2_1 * p_1_0 + 1e-7)).mean()

        return loss_softmax - self.lamb * adv_loss_non

    def forward(self, support, query, rel_text, N, K, Q, total_Q):
        """
        :param support: Inputs of the support set.
        :param query: Inputs of the query set.
        :param rel_text: Inputs of the relation description.
        :param N: Num of classes
        :param K: Num of instances for each class in the support set
        :param total_Q: Num of instances in the query set
        :return:
        """
        support_glo, support_loc = self.sentence_encoder(support)  # (B * N * K, 2D), (B * N * K, L, D)
        query_glo, query_loc = self.sentence_encoder(query)  # (B * total_Q, 2D), (B * total_Q, L, D)
        rel_text_glo, rel_text_loc = self.sentence_encoder(rel_text, cat=False)  # (B * (N + 1), D), (B * (N + 1), L, D)

        support_embedding = support_glo.view(-1, N, K, self.hidden_size * 2)  # (B, N, K, 2D)
        query_embedding = query_glo.view(-1, total_Q, self.hidden_size * 2)  # (B, total_Q, 2D)
        rel_text_loc = torch.mean(rel_text_loc, dim=1)
        rel_text_embedding = torch.cat((rel_text_glo, rel_text_loc), dim=-1)  # (B * (N + 1), 2D)
        rel_text_embedding = rel_text_embedding.view(-1, N + 1, self.hidden_size * 2)  # (B, N + 1, 2D)

        support_mean = torch.mean(support_embedding, dim=2)  # (B, N, 2D)
        logits_proto = self.__batch_dist__(support_mean, query_embedding)  # (B, total_Q, N)
        logits_relation = self.__batch_dist__(rel_text_embedding, query_embedding)  # (B, total_Q, N + 1)
        logits_relation_nota = logits_relation[:, :, N]
        logits_relation = logits_relation[:, :, :N]
        logits = (logits_proto + logits_relation) / 2
        logits = torch.cat((logits, logits_relation_nota.unsqueeze(-1)), dim=-1)  # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)

        logits_relation_for_loss = logits_relation
        logits_proto_for_loss = logits_proto

        return logits, logits_relation_for_loss, logits_proto_for_loss, pred
