import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

class FeatureProjector(nn.Module):
    def __init__(self, feat_size):
        super(FeatureProjector, self).__init__()
        self.feat_proj_f1 = nn.Linear(feat_size, feat_size)

    def forward(self, x):
        projected_feature = self.feat_proj_f1(x)
        return projected_feature
