# This script is used to plot the metrics and losses in real FsDet task

import json
import matplotlib.pyplot as plt
import numpy as np

metrics={}
path = {}
# voc
path['1shot'] = "checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['2shot'] = "checkpoints_temp/voc_split1_tfa_cos_2shot.json"
path['3shot'] = "checkpoints_temp/voc_split1_tfa_cos_3shot.json"
path['5shot'] = "checkpoints_temp/voc_split1_tfa_cos_5shot.json"
path['10shot'] = "checkpoints_temp/voc_split1_tfa_cos_10shot.json"
path['1shot LBFGS'] = "checkpoints_temp/LBFGS/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"

path['CG_L2 cg_iter_num 2'] = "checkpoints_temp/CG_L2/cg_iter2_exp0/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['CG_L2 cg_iter_num 3'] = "checkpoints_temp/CG_L2/cg_iter3_exp0/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['CG_L2 cg_iter_num 4'] = "checkpoints_temp/CG_L2/cg_iter4_exp0/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['CG_L2 cg_iter_num 5'] = "checkpoints_temp/CG_L2/cg_iter5_exp0/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['CG_L2 cg_iter_num 7'] = "checkpoints_temp/CG_L2/cg_iter7_exp0/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['CG_L2 cg_iter_num 9'] = "checkpoints_temp/CG_L2/cg_iter9_exp0/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"

path['VOC CG_L2 cg_iter_num 2 reg 0'] = "checkpoints_temp/CG_L2/voc/cg_iter2_4000_reg0/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['VOC CG_L2 cg_iter_num 2 reg 1e-4'] = "checkpoints_temp/CG_L2/voc/cg_iter2_4000_reg1e-4/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['VOC CG_L2 cg_iter_num 2 reg 1e-2'] = "checkpoints_temp/CG_L2/voc/cg_iter2_4000_reg1e-2/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['VOC CG_L2 cg_iter_num 2 reg 1'] = "checkpoints_temp/CG_L2/voc/cg_iter2_4000_reg1/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['VOC CG_L2 cg_iter_num 3 reg 0'] = "checkpoints_temp/CG_L2/voc/cg_iter3_4000_reg0/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['VOC CG_L2 cg_iter_num 3 reg 1e-4'] = "checkpoints_temp/CG_L2/voc/cg_iter3_4000_reg1e-4/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['VOC CG_L2 cg_iter_num 3 reg 1e-2'] = "checkpoints_temp/CG_L2/voc/cg_iter3_4000_reg1e-2/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['VOC CG_L2 cg_iter_num 3 reg 1'] = "checkpoints_temp/CG_L2/voc/cg_iter3_4000_reg1/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"

path['SGD_L1 Feature Extracted'] = "checkpoints_temp/SGD_L1_modified/exp_0/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['SGD_L2 Feature Extracted'] = "checkpoints_temp/SGD_L2_modified/exp_0/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['SGD_L1 Original'] = "checkpoints_temp/SGD_L1/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['SGD_L2 Original'] = "checkpoints_temp/SGD_L2/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"

path['VOC SGD_L2 Feature Extracted Mask'] = "checkpoints_temp/SGD_L2_modified_mask/exp_0/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"
path['VOC SGD_L2 Feature Extracted No Mask'] = "checkpoints_temp/SGD_L2_modified_no_mask/exp_0/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/metrics.json"

# coco
path['COCO SGD_L1 Original'] = "checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/metrics.json"
path['COCO SGD_L2 Feature Extracted'] = "checkpoints_temp/SGD_L2_modified/coco/exp_0/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/metrics.json"
path['COCO CG_L2 cg_iter_num 2'] = "checkpoints_temp/CG_L2/coco/cg_iter2_exp0/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/metrics.json"
path['COCO CG_L2 cg_iter_num 3'] = "checkpoints_temp/CG_L2/coco/cg_iter3_exp0/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/metrics.json"

# coco rigid two stage fc
path['COCO rigid_two_stage fc randinit']='checkpoints_temp/CG_L2_test/coco/cg2_shot1_iter200_rts_fc_randinit/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/metrics.json'
path['COCO rigid_two_stage fc combine']='checkpoints_temp/CG_L2_test/coco/cg2_shot1_iter200_rts_fc_combine/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/metrics.json'

# coco fc RTS filter_3levels+bg
path['RTS SGD_L2 iter_1000 shot_1 randinit']='checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_iter1000_rts_randinit/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/metrics.json'
path['Novel SGD_L2 iter_500 shot_1']='checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_novel_iter500/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/metrics.json'

path['RTS CG_L2 iter_100 shot_1 randinit test1']='checkpoints_temp/CG_L2_test/coco/cg2_shot1_iter100_rts_fc_randinit_debug_filter_3levels_bg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/metrics.json'
path['RTS CG_L2 iter_60 shot_1 randinit test2']='checkpoints_temp/CG_L2_test/coco/cg2_shot1_iter60_rts_fc_randinit_debug_filter_3levels_bg_test2/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/metrics.json'
path['Novel CG_L2 iter_200 shot_1']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter200/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/metrics.json'

path['RTS CG_L2 iter_200 shot_1 randinit reg_1e-1']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter200_reg1e-1/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/metrics.json'
path['RTS CG_L2 iter_200 shot_1 randinit reg_1']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter200_reg1/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/metrics.json'
path['RTS CG_L2 iter_100 shot_1 randinit grad_clip_norm_1']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter100_gradclip_norm1/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/metrics.json'

# coco meta rts novel weight loss
# SGD
path['Novel SGD_L2 iter_500 shot_1 seed_0']="checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_novel_iter500_seed0/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/metrics.json"
path['Novel SGD_L2 iter_500 shot_1 seed_10']="checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_novel_iter500_seed10/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/metrics.json"
path['Novel SGD_L2 iter_500 shot_1 meta_predweight_tfa']="checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_novel_iter500_metaweight_test/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/metrics.json"
path['Novel SGD_L2 iter_500 shot_1 meta_predweight_novel']="checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_novel_iter500_metaweight_novel_newton0_iter1000_exp1/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/metrics.json"
path['Novel SGD_L2 iter_500 shot_1 meta_initweight_novel']="checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_novel_iter500_metainitweight_novel_newton0_iter1000/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/metrics.json"
# CG
path['Novel CG_L2 iter_20 shot_1 sgd_weight_iter100_seed0']="checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter20_sgdweightseed0_novel_iter100/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/metrics.json"
path['Novel CG_L2 iter_20 shot_1 sgd_weight_iter100_seed10']="checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter20_sgdweightseed10_novel_iter100/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/metrics.json"
path['Novel CG_L2 iter_20 shot_1 meta_predweight_tfa']="checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter20_metaweight_test/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/metrics.json"
path['Novel CG_L2 iter_20 shot_1 meta_predweight_novel']="checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter20_metaweight_novel_newton0_iter1000/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/metrics.json"
path['Novel CG_L2 iter_20 shot_1 meta_initweight_novel']="checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter20_metainitweight_novel_newton0_iter1000/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/metrics.json"



x='Novel SGD_L2 iter_500 shot_1 seed_0'
x='Novel SGD_L2 iter_500 shot_1 seed_10'
x='Novel SGD_L2 iter_500 shot_1 meta_predweight_tfa'
x='Novel SGD_L2 iter_500 shot_1 meta_predweight_novel'
x='Novel SGD_L2 iter_500 shot_1 meta_initweight_novel'

x='Novel CG_L2 iter_20 shot_1 sgd_weight_iter100_seed0'
x='Novel CG_L2 iter_20 shot_1 sgd_weight_iter100_seed10'
x='Novel CG_L2 iter_20 shot_1 meta_predweight_tfa'
x='Novel CG_L2 iter_20 shot_1 meta_predweight_novel'
x='Novel CG_L2 iter_20 shot_1 meta_initweight_novel'

metrics[x]=[]
with open(path[x],'r') as f:
    for line in f.readlines():
        metrics[x].append(json.loads(line))

losses ={}
losses[x]={}
# magnitude = float(x.split(' ')[-1])
magnitude = 0
loss_names = ['total_loss', 'loss_cls', 'loss_box_reg', 'loss_weight'] if magnitude!=0 else ['total_loss', 'loss_cls', 'loss_box_reg']
for loss_name in loss_names:
    if loss_name == 'loss_weight':
        losses[x][loss_name] = [i[loss_name]/magnitude for i in metrics[x][:-1]]
    else:
        losses[x][loss_name] = [i[loss_name] for i in metrics[x][:-1]]

xAxis_ticks={}
xAxis_ticks[x] = [i['iteration'] for i in metrics[x][:-1]]
y_lims = {'total_loss': [0, 3.5], 'loss_cls': [0, 3.5], 'loss_box_reg': [0, 0.16], 'loss_weight': [2, 18.5]}

fig, axs = plt.subplots(4,1, figsize=(6,14.4)) if magnitude!=0 else plt.subplots(3,1, figsize=(6,10.8))
for idx, loss_name in enumerate(loss_names):
    axs[idx].plot(xAxis_ticks[x], losses[x][loss_name], label=loss_name)
    axs[idx].set_ylim(y_lims[loss_name])
    axs[idx].legend(loc='upper right', fontsize=12)
axs[0].set_title('{}'.format(x))
axs[-1].set_xlabel('num of iteration', fontsize=12)

plt.tight_layout()
plt.savefig('checkpoints_temp/All_figs/June30_figs/loss_{}.png'. format(x))

# ## Print the model to check frozen params
# #%% 
# from fsdet.model_zoo import get_config_file
# from fsdet.config import get_cfg
# from fsdet.modeling import build_model
# import torch
# #%%
# config_path = "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml"
# cfg_file = get_config_file(config_path)

# cfg = get_cfg()
# cfg.merge_from_file(cfg_file)

# if not torch.cuda.is_available():
#     cfg.MODEL.DEVICE = "cpu"

# model = build_model(cfg)
# # %%
