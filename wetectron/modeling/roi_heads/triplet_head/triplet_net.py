import torch
import torch.nn as nn
import torch.nn.functional as F
#import fvcore.nn.weight_init as weight_init
from wetectron.modeling.poolers import Pooler
#from wetectron.modeling.dropblock.drop_block import DropBlock2D, Drop_Seq

class Sim_Net(nn.Module):
    def __init__(self, config, in_dim):
        super(Sim_Net, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,128)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #(w, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, roi_feat):
        return F.normalize(self.mlp(roi_feat), dim=1)
