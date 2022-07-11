import torch
from torch import nn
from wetectron.utils.tools import cos_sim
from torch.nn import functional as F

class Dist_Loss(nn.Module):
    def __init__(self):
        self.cos_lmda = cfg.
        self.clr_lmda = cfg.
        self.cos_target =

    def forward(self, a_feat, p_feat, n_feat, ):
        clr_lmda = source_score[torch.cat((a_overlaps[i], p_overlaps[i] + p_size)),duplicate].mean()
        if n_feat.nelement() != 0:
            return_loss_dict['loss_clr%d'%i] = self.cos_lmda * (F.relu(cos_sim(a_feat, n_feat).mean() - cos_sim(a_feat, p_feat).mean() + self.margin)\
                                                 + F.relu(cos_sim(p_feat, n_feat).mean() - cos_sim(a_feat, p_feat).mean() + self.margin))/2
        else :
            return_loss_dict['loss_clr%d'%i] = self.cos_lmda * F.relu(cos_sim(a_feat, p_feat).mean())
        return return_loss_dict['loss_clr%d']

class Clr_Loss(nn.Module):
    def __init__(self):

    def forward():
        return
