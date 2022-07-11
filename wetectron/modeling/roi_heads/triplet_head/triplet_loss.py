from torch.nn import functional as F
from wetectron.utils.utils import cos_sim
import torch.nn as nn
import torch
from itertools import combinations

class Triplet_Loss(nn.Module):
    def __init__(self):
        super(Triplet_Loss, self).__init__()
        self.margin = 0.5

    def forward(self, anchor, pos, n_feat, b1_n_pgt, b2_n_pgt, b1_n_pgt_cat, b2_n_pgt_cat, p_size, sim_feature):
    #def forward(self, anchor, pos, neg, score_weight):
        '''if neg.nelement() != 0:
            cos_loss = score_weight * (F.relu(cos_sim(anchor, neg).mean() - cos_sim(anchor, pos).mean() + self.margin)\
                                    + F.relu(cos_sim(pos, neg).mean() - cos_sim(anchor, pos).mean() + self.margin))/2
        else:
            cos_loss = score_weight * F.relu(1 - cos_sim(anchor, pos).mean())
        '''

        cos_loss = torch.zeros((0), dtype=torch.float, device=sim_feature.device)
        b2_n_pgt = [b2_neg_pgt + p_size for b2_neg_pgt in b2_n_pgt]
        n_pgt = b1_n_pgt + b2_n_pgt
        #n_pair = list(combinations(n_pgt,2))
        if len(n_pgt) != 0:
            for n in n_pgt:
                cos_loss = torch.cat(( cos_loss, F.relu(1 + cos_sim(anchor, sim_feature[n]).mean()).view(-1) ))
                cos_loss = torch.cat(( cos_loss, F.relu(1 + cos_sim(pos, sim_feature[n]).mean()).view(-1) ))
            cos_loss = torch.cat(( cos_loss, F.relu(1 - cos_sim(anchor,pos).mean()).view(-1) ))
            #if n_pair:
            #    for n_p in n_pair:
            #        cos_loss = torch.cat(( cos_loss, F.relu(1 + cos_sim(sim_feature[n_p[0]], sim_feature[n_p[1]]).mean()).view(-1) ))
            cos_loss = cos_loss.mean()
        else :
            cos_loss = F.relu(1 - cos_sim(anchor,pos).mean())
        return cos_loss

class N_pair_Loss(nn.Module):
    def __init__(self):
        super(N_pair_Loss, self).__init__()
    #def forward(self, anchor, pos, n_feat, b1_n_pgt, b2_n_pgt, b1_n_pgt_cat, b2_n_pgt_cat, p_size, sim_feature):
    def forward(self, overlaps_enc, device):

        for i, embedding_v in enumerate(overlaps_enc):
            if embedding_v.nelement() != 0:
                import IPython; IPython.embed()

        b2_n_pgt = [b2_neg_pgt + p_size for b2_neg_pgt in b2_n_pgt]
        n_pgt = b1_n_pgt + b2_n_pgt
        ap_sim = cos_sim(anchor, pos).mean()
        exp_sum = torch.zeros((), dtype=torch.float, device=sim_feature.device)
        if len(n_pgt) != 0:
            for n in n_pgt:
                exp_sum += torch.exp(cos_sim(anchor, sim_feature[n]).mean() - ap_sim)
        else:
            exp_sum = torch.exp(-ap_sim)
        npair_loss = torch.log(1+exp_sum)
        return npair_loss

class Contra_Loss(nn.Module):
    def __init__(self):
        super(Contra_Loss, self).__init__()
        self.margin = 0.5
        self.lmda = 0.01

    def forward(self, overlaps_enc, score_weight, device):

        features = torch.zeros((0), dtype=torch.float, device=device)
        labels = torch.zeros((0), dtype=torch.float, device=device)
        for i, embedding_v in enumerate(overlaps_enc):
            if embedding_v.shape[0] != 0:
                features = torch.cat((features, embedding_v))
                labels = torch.cat((labels, torch.ones(embedding_v.shape[0]).to(device)*(i)))

        import IPython; IPython.embed()
        return self.lmda * cos_loss.mean()

class Supcon_Loss(nn.Module):
    def __init__(self, temp):
        super(Supcon_Loss, self).__init__()
        self.temperature = temp
    def forward(self, overlaps_enc, device):
    #def forward(self, overlaps_enc, score_col, device):
        features = torch.zeros((0), dtype=torch.float, device=device)
        labels = torch.zeros((0), dtype=torch.float, device=device)
        #score_weight = score_col.detach()
        #mask = torch.eq(labels.view(-1,1), labels.view(-1,1).T).float().cuda()
        import IPython; IPython.embed()
        ### custom supervision from overlaps_encoding ###
        for i, embedding_v in enumerate(overlaps_enc):
            if embedding_v.shape[0] != 0:
                features = torch.cat((features, embedding_v))
                labels = torch.cat((labels, torch.ones(embedding_v.shape[0]).to(device)*(i+1)))
        ### same as supcon loss ###

        similarity = torch.div(torch.mm(features, features.T), self.temperature)
        #similarity = torch.mul(torch.mm(features, features.T), score_weight)                  ### temp_score

        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        label_mask = torch.eq(labels.view(-1,1), labels.view(-1,1).T).float().cuda()
        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)


        loss = - per_label_log_prob
        #loss = - per_label_log_prob * score_weight                                             ### contral weight
        return loss.mean()

class SupConLossV2(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, overlaps_enc, score_col, device):
    #def forward(self, overlaps_enc, device):
        features = torch.zeros((0), dtype=torch.float, device=device)
        labels = torch.zeros((0), dtype=torch.float, device=device)
        score_weight = score_col.detach()
        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        #label_mask = torch.eq(labels, labels.T).float().cuda()

        for i, embedding_v in enumerate(overlaps_enc):
            if embedding_v.shape[0] != 0:
                features = torch.cat((features, embedding_v))
                labels = torch.cat((labels, torch.ones(embedding_v.shape[0]).to(device)*(i)))
        #import IPython; IPython.embed()
        similarity = torch.div(torch.matmul(features, features.T), self.temperature)

        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)


        exp_sim = torch.exp(similarity)
        label_mask = torch.eq(labels.view(-1,1), labels.view(-1,1).T).float().cuda()

        mask = logits_mask * label_mask

        log_prob = torch.log( (exp_sim * mask).sum(1) / (exp_sim * logits_mask).sum(1) )
        #log_prob = torch.log( (exp_sim * mask * score_weight).sum(1) / (exp_sim * logits_mask * score_weight).sum(1) )

        #loss = -log_prob
        loss = -log_prob * score_weight

        return loss.mean()

