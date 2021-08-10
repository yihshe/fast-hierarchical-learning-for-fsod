import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from pytracking.libs.tensorlist import TensorList

# NOTE currently the weight predictor module is only designed for CosineOutputLayers
class WeightPredictor(nn.Module):
    def __init__(self, feat_size, box_dim=4):
        super(WeightPredictor, self).__init__()

        self.cls_weight_pred_f1 = nn.Linear(feat_size, feat_size)
        self.cls_weight_pred_f2 = nn.Linear(feat_size, feat_size)

        # NOTE rts
        self.cls_bias_pred_f1 = nn.Linear(feat_size, feat_size)
        self.cls_bias_pred_f2 = nn.Linear(feat_size, 1)

        self.cls_score_weight_bg = nn.Parameter(torch.rand(feat_size).view(-1, feat_size))
        torch.nn.init.normal_(self.cls_score_weight_bg, 0, 0.01)
        self.cls_score_bias_bg = nn.Parameter(torch.zeros(1))

        self.bbox_weight_pred_f1 = nn.Linear(feat_size, feat_size*box_dim)
        self.bbox_weight_pred_f2 = nn.Linear(feat_size*box_dim, feat_size*box_dim)

        # NOTE rts
        # self.bbox_bias_pred_f1 = nn.Linear(feat_size, box_dim)
        # self.bbox_bias_pred_f2 = nn.Linear(box_dim, box_dim)
        self.bbox_bias_pred_f1 = nn.Linear(feat_size, feat_size)
        self.bbox_bias_pred_f2 = nn.Linear(feat_size, box_dim)

        self.feat_size = feat_size

        self.scale_factor = 0.1

    def forward(self, x):
        cls_score_weight = self._forward_cls_weight_pred(x)
        cls_score_bias = self._forward_cls_bias_pred(x).view(-1)
        bbox_pred_weight = self._forward_bbox_weight_pred(x)
        bbox_pred_bias = self._forward_bbox_bias_pred(x).view(-1)
        # NOTE tfa
        # return TensorList([cls_score_weight, cls_score_bias, bbox_pred_weight, bbox_pred_bias])*self.scale_factor
        
        # NOTE rts
        pred_weights = TensorList([cls_score_weight, cls_score_bias, bbox_pred_weight, bbox_pred_bias])*self.scale_factor
        pred_weights[0] = torch.cat((pred_weights[0], self.cls_score_weight_bg), dim=0)
        pred_weights[1] = torch.cat((pred_weights[1], self.cls_score_bias_bg), dim=0)
        return pred_weights

    def _forward_cls_weight_pred(self, x):
        x = F.relu(self.cls_weight_pred_f1(x))
        x = self.cls_weight_pred_f2(x)
        # x = self.cls_weight_pred_f1(x)

        return x

    def _forward_cls_bias_pred(self, x):
        x = F.relu(self.cls_bias_pred_f1(x))
        x = self.cls_bias_pred_f2(x)

        return x

    def _forward_bbox_weight_pred(self, x):
        x = F.relu(self.bbox_weight_pred_f1(x))
        x = self.bbox_weight_pred_f2(x)
        # x = self.bbox_weight_pred_f1(x)

        x = x.view(-1, self.feat_size)
        return x

    def _forward_bbox_bias_pred(self, x):
        x = F.relu(self.bbox_bias_pred_f1(x))
        x = self.bbox_bias_pred_f2(x)
        # x = self.bbox_bias_pred_f1(x)

        x = x.view(-1)
        return x

class WeightPredictor2(nn.Module):
    def __init__(self, feat_size, box_dim=4):
        super(WeightPredictor2, self).__init__()
        self.cls_score_weight = nn.Parameter(torch.rand(21, feat_size))
        torch.nn.init.normal_(self.cls_score_weight, 0, 0.01)
        self.cls_score_bias = nn.Parameter(torch.zeros(21))

        self.bbox_pred_weight = nn.Parameter(torch.rand(80, feat_size))
        torch.nn.init.normal_(self.bbox_pred_weight, 0, 0.01)
        self.bbox_pred_bias = nn.Parameter(torch.zeros(80))

    def forward(self, x):
        return TensorList([self.cls_score_weight, self.cls_score_bias, self.bbox_pred_weight, self.bbox_pred_bias])

class WeightPredictor3(nn.Module):
    def __init__(self, feat_size, box_dim=4):
        super(WeightPredictor3, self).__init__()

        self.cls_weight_pred_f1 = nn.Linear(feat_size, feat_size)
        self.cls_weight_pred_f2 = nn.Linear(feat_size, feat_size)

        self.bbox_weight_pred_f1 = nn.Linear(feat_size, feat_size*box_dim)
        self.bbox_weight_pred_f2 = nn.Linear(feat_size*box_dim, feat_size*box_dim)

        self.bbox_bias_pred_f1 = nn.Linear(feat_size, box_dim)
        self.bbox_bias_pred_f2 = nn.Linear(box_dim, box_dim)

        self.feat_size = feat_size

        self.scale_factor = 0.1

    def forward(self, x):
        cls_score_weight = self._forward_cls_weight_pred(x)
        bbox_pred_weight = self._forward_bbox_weight_pred(x)
        bbox_pred_bias = self._forward_bbox_bias_pred(x).view(-1)
        return TensorList([cls_score_weight, bbox_pred_weight, bbox_pred_bias])*self.scale_factor

    def _forward_cls_weight_pred(self, x):
        x = F.relu(self.cls_weight_pred_f1(x))
        x = self.cls_weight_pred_f2(x)
        # x = self.cls_weight_pred_f1(x)

        return x

    def _forward_bbox_weight_pred(self, x):
        x = F.relu(self.bbox_weight_pred_f1(x))
        x = self.bbox_weight_pred_f2(x)
        # x = self.bbox_weight_pred_f1(x)

        x = x.view(-1, self.feat_size)
        return x

    def _forward_bbox_bias_pred(self, x):
        x = F.relu(self.bbox_bias_pred_f1(x))
        x = self.bbox_bias_pred_f2(x)
        # x = self.bbox_bias_pred_f1(x)

        x = x.view(-1)
        return x


