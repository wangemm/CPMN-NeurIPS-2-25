import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys


def wmse_loss(input, target, weight, reduction='mean'):
    ret = (torch.diag(weight).mm(target - input)) ** 2
    ret = torch.mean(ret)
    return ret


class PCLoss(nn.Module):
    def __init__(self, args):
        super(PCLoss, self).__init__()
        self.t = args.t
        self.topk = args.topk
        self.device = args.device
        self.normalize = args.normalize_loss

    def graph_loss(self, z0, z1):

        z0 = F.normalize(z0, dim=-1)
        z1 = F.normalize(z1, dim=-1)

        sv_sim0 = torch.mm(z0, z0.T) / self.t
        sv_sim1 = torch.mm(z1, z1.T) / self.t

        def get_mutual_mask(sim_matrix):
            _, topk_idx = torch.topk(sim_matrix, self.topk * 2, dim=1)
            adj_matrix = torch.zeros_like(sim_matrix, dtype=torch.bool)
            adj_matrix.scatter_(1, topk_idx, True)
            mutual_mask = adj_matrix & adj_matrix.T
            no_pos_mask = (mutual_mask.sum(1) == 0)
            if no_pos_mask.any():
                print(f"Warning: {no_pos_mask.sum()} samples have no mutual neighbors")
                mutual_mask[no_pos_mask] = adj_matrix[no_pos_mask]
            return mutual_mask.float()

        pos_labels0 = get_mutual_mask(sv_sim0.masked_fill(torch.eye(z0.size(0), device=self.device).bool(), -1e9))
        pos_labels1 = get_mutual_mask(sv_sim1.masked_fill(torch.eye(z1.size(0), device=self.device).bool(), -1e9))

        logits0 = sv_sim0 / self.t
        log_prob0 = F.log_softmax(logits0, dim=1)
        pos_count0 = pos_labels0.sum(1, keepdim=True).clamp(min=1e-8)
        loss0 = -(pos_labels0 * log_prob0).sum(1) / pos_count0.squeeze()

        logits1 = sv_sim1 / self.t
        log_prob1 = F.log_softmax(logits1, dim=1)
        pos_count1 = pos_labels1.sum(1, keepdim=True).clamp(min=1e-8)
        loss1 = -(pos_labels1 * log_prob1).sum(1) / pos_count1.squeeze()

        loss_single = (loss0.mean() + loss1.mean()) / 2

        cross_sim = torch.mm(z0, z1.t())
        _, topk_a2b = torch.topk(cross_sim, self.topk * 2, dim=1)  # A->B
        _, topk_b2a = torch.topk(cross_sim, self.topk * 2, dim=0)  # B->A

        adj_a2b = torch.zeros_like(cross_sim, dtype=torch.bool)
        adj_a2b.scatter_(1, topk_a2b, True)

        adj_b2a = torch.zeros_like(cross_sim, dtype=torch.bool)
        adj_b2a.scatter_(0, topk_b2a, True)

        mutual_cross = adj_a2b & adj_b2a
        pos_labels = mutual_cross.float()

        logits = cross_sim / self.t
        log_prob = F.log_softmax(logits, dim=1)
        total_pos = pos_labels.sum().clamp(min=1e-8)
        loss_cross = -(pos_labels * log_prob).sum() / total_pos

        loss = (loss_single + loss_cross) / 2
        return loss

    def prototype_loss(self, z, c, c_cross, y, y_cross):

        if self.normalize:
            z = F.normalize(z, p=2, dim=1)
            c = F.normalize(c, p=2, dim=1)
            c_cross = F.normalize(c_cross, p=2, dim=1)

        prototype_logits = torch.mm(z, c.T) / self.t
        prototype_logits_c = torch.mm(z, c_cross.T) / self.t

        prototype_loss = (F.cross_entropy(prototype_logits, y) + F.cross_entropy(prototype_logits_c, y)
                          + F.cross_entropy(prototype_logits, y_cross) + F.cross_entropy(prototype_logits_c, y_cross))

        sample_logits = torch.mm(z, z.T) / self.t
        sample_log_prob = F.log_softmax(sample_logits, dim=1)

        sample_labels = (y.unsqueeze(0) == y.unsqueeze(1)).float()
        sample_mean_log_prob_pos = (sample_labels * sample_log_prob).sum(1) / (sample_labels.sum(1) + 1e-8)
        sample_loss = -sample_mean_log_prob_pos.mean()

        sample_labels_c = (y_cross.unsqueeze(0) == y_cross.unsqueeze(1)).float()
        sample_mean_log_prob_pos_c = (sample_labels_c * sample_log_prob).sum(1) / (sample_labels_c.sum(1) + 1e-8)
        sample_loss_c = -sample_mean_log_prob_pos_c.mean()

        sample_loss_a = sample_loss + sample_loss_c

        total_loss = prototype_loss + sample_loss_a
        return total_loss
