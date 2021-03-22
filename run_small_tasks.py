## Download base model
# from fsdet import model_zoo
# import torch
# model_base = model_zoo.get("PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml", trained=True)
# torch.save(model_base.state_dict(), 'checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth')
# print(model_base)

# Plot metrics for fine tuning
#%%
import json
import matplotlib.pyplot as plt
import numpy as np
#%%
metrics={}
path = {}
path['1shot'] = "checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['2shot'] = "checkpoints_temp/voc_split1_tfa_cos_2shot.json"
path['3shot'] = "checkpoints_temp/voc_split1_tfa_cos_3shot.json"
path['5shot'] = "checkpoints_temp/voc_split1_tfa_cos_5shot.json"
path['10shot'] = "checkpoints_temp/voc_split1_tfa_cos_10shot.json"
path['1shot LBFGS'] = "checkpoints_temp/LBFGS/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
x = '1shot LBFGS'
metrics[x]=[]
with open(path[x],'r') as f:
    for line in f.readlines():
        metrics[x].append(json.loads(line))
# %%
losses ={}
losses[x]={}
losses[x]['total_loss'] = [i['total_loss'] for i in metrics[x][:-1]]
losses[x]['loss_box_reg'] = [i['loss_box_reg'] for i in metrics[x][:-1]]
losses[x]['loss_cls'] = [i['loss_cls'] for i in metrics[x][:-1]]
losses[x]['loss_rpn_cls'] = [i['loss_rpn_cls'] for i in metrics[x][:-1]]
losses[x]['loss_rpn_loc'] = [i['loss_rpn_loc'] for i in metrics[x][:-1]]

# %%
xAxis_ticks={}
xAxis_ticks[x] = [(i+1)*20 for i in range(len(metrics[x][:-1]))]
fig, axs = plt.subplots(5,1, figsize=(10,20))
axs[0].plot(xAxis_ticks[x], losses[x]['total_loss'], label='total_loss')
axs[1].plot(xAxis_ticks[x], losses[x]['loss_box_reg'], label='loss_box_reg')
axs[2].plot(xAxis_ticks[x], losses[x]['loss_cls'], label='loss_cls')
axs[3].plot(xAxis_ticks[x], losses[x]['loss_rpn_cls'], label='loss_rpn_cls')
axs[4].plot(xAxis_ticks[x], losses[x]['loss_rpn_loc'], label='loss_rpn_loc')
axs[4].set_xlabel('num of iteration', fontsize=12)
for ax in axs:
    ax.legend(loc='upper right', fontsize=12)
fig.suptitle('PASCAL VOC Split1 {}'.format(x), x=0.5, y=0.9)
# plt.savefig('checkpoints_temp/{}.png'. format(x))

## Print the model to check frozen params
#%% 
from fsdet.model_zoo import get_config_file
from fsdet.config import get_cfg
from fsdet.modeling import build_model
import torch
#%%
config_path = "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml"
cfg_file = get_config_file(config_path)

cfg = get_cfg()
cfg.merge_from_file(cfg_file)

if not torch.cuda.is_available():
    cfg.MODEL.DEVICE = "cpu"

model = build_model(cfg)
# %%
