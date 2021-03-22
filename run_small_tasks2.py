from fsdet.model_zoo import get_config_file
from fsdet.config import get_cfg
from fsdet.modeling import build_model
import torch
from IPython import embed

config_path = "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml"
cfg_file = get_config_file(config_path)

cfg = get_cfg()
cfg.merge_from_file(cfg_file)

if not torch.cuda.is_available():
    cfg.MODEL.DEVICE = "cpu"

model = build_model(cfg)
params = [param for param in model.parameters()]

embed()
