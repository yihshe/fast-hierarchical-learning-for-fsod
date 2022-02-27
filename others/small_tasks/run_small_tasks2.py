# This script is used to plot metrics for saved checkpoints in real FsDet task

import json
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

# metrics={}
path = {}
# ckpt_keys = {}
# voc
path['CG_L2 cg_iter_num_2 newton_iter_num_4000'] = "checkpoints_temp/CG_L2/cg_iter2_exp0/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"
path['CG_L2 cg_iter_num_3 newton_iter_num_4000'] = "checkpoints_temp/CG_L2/cg_iter3_exp0/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"
# path['SGD_L2 feature_extracted num_iter_4000'] = "checkpoints_temp/SGD_L2_modified/exp_0/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"
path['SGD_L2 feature_extracted num_iter_4000'] = "checkpoints_temp/SGD_L2_modified/voc/exp_3_reg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"

path['CG_L2 cg_iter_num_2 newton_iter_num_4000 reg0'] = "checkpoints_temp/CG_L2/voc/cg_iter2_4000_reg0/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"
path['CG_L2 cg_iter_num_2 newton_iter_num_4000 reg1e-4'] = "checkpoints_temp/CG_L2/voc/cg_iter2_4000_reg1e-4/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"
path['CG_L2 cg_iter_num_2 newton_iter_num_4000 reg1e-2'] = "checkpoints_temp/CG_L2/voc/cg_iter2_4000_reg1e-2/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"
path['CG_L2 cg_iter_num_2 newton_iter_num_4000 reg1'] = "checkpoints_temp/CG_L2/voc/cg_iter2_4000_reg1/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"
path['CG_L2 cg_iter_num_2 newton_iter_num_500 reg1e-2'] = "checkpoints_temp/CG_L2/voc/cg_iter2_500_reg1e-2/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"

path['CG_L2 cg_iter_num_3 newton_iter_num_4000 reg0'] = "checkpoints_temp/CG_L2/voc/cg_iter3_4000_reg0/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"
path['CG_L2 cg_iter_num_3 newton_iter_num_4000 reg1e-4'] = "checkpoints_temp/CG_L2/voc/cg_iter3_4000_reg1e-4/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"
path['CG_L2 cg_iter_num_3 newton_iter_num_4000 reg1e-2'] = "checkpoints_temp/CG_L2/voc/cg_iter3_4000_reg1e-2/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"
path['CG_L2 cg_iter_num_3 newton_iter_num_4000 reg1'] = "checkpoints_temp/CG_L2/voc/cg_iter3_4000_reg1/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"
path['CG_L2 cg_iter_num_3 newton_iter_num_500 reg1e-2'] = "checkpoints_temp/CG_L2/voc/cg_iter3_500_reg1e-2/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"

path['CG_L2 cg_iter_num_2 VOC newton_iter_num_200 reg1e-2'] = "checkpoints_temp/CG_L2/voc/cg_iter2_200_reg1e-2/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"
path['CG_L2 cg_iter_num_2 VOC newton_iter_num_200 reg5e-3'] = "checkpoints_temp/CG_L2/voc/cg_iter2_200_reg5e-3/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"
path['CG_L2 cg_iter_num_2 VOC newton_iter_num_200 reg1e-3'] = "checkpoints_temp/CG_L2/voc/cg_iter2_200_reg1e-3/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"

# x_list = ['SGD_L2 feature_extracted num_iter_4000', 'CG_L2 cg_iter_num_2 newton_iter_num_4000 reg0', 'CG_L2 cg_iter_num_3 newton_iter_num_4000 reg0']
# x_list = ['SGD_L2 feature_extracted num_iter_4000', 'CG_L2 cg_iter_num_2 newton_iter_num_4000 reg1e-4', 'CG_L2 cg_iter_num_3 newton_iter_num_4000 reg1e-4']
# x_list = ['SGD_L2 feature_extracted num_iter_4000', 'CG_L2 cg_iter_num_2 newton_iter_num_4000 reg1e-2', 'CG_L2 cg_iter_num_3 newton_iter_num_4000 reg1e-2']
# x_list = ['SGD_L2 feature_extracted num_iter_4000', 'CG_L2 cg_iter_num_2 newton_iter_num_4000 reg1', 'CG_L2 cg_iter_num_3 newton_iter_num_4000 reg1']
# x_list = ['CG_L2 cg_iter_num_2 newton_iter_num_500 reg1e-2', 'CG_L2 cg_iter_num_3 newton_iter_num_500 reg1e-2']

# x_list = ['CG_L2 cg_iter_num_2 VOC newton_iter_num_200 reg1e-2', 'CG_L2 cg_iter_num_2 VOC newton_iter_num_200 reg5e-3', 'CG_L2 cg_iter_num_2 VOC newton_iter_num_200 reg1e-3']

path['CG_L2 cg_iter_num_2 newton_iter_num_500'] = "checkpoints_temp/CG_L2/cg_iter2_exp1_500/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"
path['CG_L2 cg_iter_num_3 newton_iter_num_500'] = "checkpoints_temp/CG_L2/cg_iter3_exp2_500/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"

path['CG_L2 cg_iter_num_2 newton_iter_num_50'] = "checkpoints_temp/CG_L2/cg_iter2_exp2_50/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"
path['CG_L2 cg_iter_num_3 newton_iter_num_50'] = "checkpoints_temp/CG_L2/cg_iter3_exp3_50/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel/inference/all_res.json"

# coco
path['CG_L2 cg_iter_num_2 newton_iter_num_16000'] = "checkpoints_temp/CG_L2/coco/cg_iter2_exp0/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['CG_L2 cg_iter_num_3 newton_iter_num_16000'] = "checkpoints_temp/CG_L2/coco/cg_iter3_exp0/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
# path['SGD_L2 feature_extracted num_iter_16000'] = "checkpoints_temp/SGD_L2_modified/coco/exp_0/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['SGD_L2 feature_extracted COCO num_iter_16000'] = "checkpoints_temp/SGD_L2_modified/coco/exp_1_reg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['SGD_L2 feature_full_batch COCO num_iter_16000'] = "checkpoints_temp/SGD_L2_modified/coco/exp_2_reg_fullFeat/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"

path['CG_L2 cg_iter_num_2 newton_iter_num_1000'] = "checkpoints_temp/CG_L2/coco/cg_iter2_exp1_1000/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['CG_L2 cg_iter_num_3 newton_iter_num_1000'] = "checkpoints_temp/CG_L2/coco/cg_iter3_exp1_1000/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"

path['CG_L2 cg_iter_num_2 newton_iter_num_100'] = "checkpoints_temp/CG_L2/coco/cg_iter2_exp2_100/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['CG_L2 cg_iter_num_3 newton_iter_num_100'] = "checkpoints_temp/CG_L2/coco/cg_iter3_exp2_100/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"

path['CG_L2 cg_iter_num_2 COCO newton_iter_num_200 reg1e-2'] = "checkpoints_temp/CG_L2/coco/cg_iter2_exp200_reg1e-2/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['CG_L2 cg_iter_num_2 COCO newton_iter_num_200 reg5e-3'] = "checkpoints_temp/CG_L2/coco/cg_iter2_exp200_reg5e-3/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['CG_L2 cg_iter_num_2 COCO newton_iter_num_200 reg1e-3'] = "checkpoints_temp/CG_L2/coco/cg_iter2_exp200_reg1e-3/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"

# coco more shots
path['CG_L2 cg_iter_2 newton_iter_200 shot_1']='checkpoints_temp/CG_L2/coco/cg2_shot1_iter200/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['CG_L2 cg_iter_2 newton_iter_400 shot_1']='checkpoints_temp/CG_L2/coco/cg2_shot1_iter400/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'

path['CG_L2 cg_iter_2 newton_iter_400 shot_2']='checkpoints_temp/CG_L2/coco/cg2_shot2_iter400/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot/inference/all_res.json'
path['CG_L2 cg_iter_2 newton_iter_400 shot_3']='checkpoints_temp/CG_L2/coco/cg2_shot3_iter400/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot/inference/all_res.json'
path['CG_L2 cg_iter_2 newton_iter_400 shot_5']='checkpoints_temp/CG_L2/coco/cg2_shot5_iter400/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot/inference/all_res.json'

# x_list = ['CG_L2 cg_iter_2 newton_iter_400 shot_5']

#meta
path['coco_meta cg_iter_2 newton_iter_5 lambda_learned 1'] = "checkpoints_temp/CG_L2/coco/meta_reg1_iter500_learned/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['coco_meta cg_iter_2 newton_iter_5 lambda_init 1'] = "checkpoints_temp/CG_L2/coco/meta_reg1_iter500_init/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['coco_meta cg_iter_2 newton_iter_5 lambda_learned 5e-1'] = "checkpoints_temp/CG_L2/coco/meta_reg5e-1_iter500_learned/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['coco_meta cg_iter_2 newton_iter_5 lambda_init 5e-1'] = "checkpoints_temp/CG_L2/coco/meta_reg5e-1_iter500_init/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['coco_meta cg_iter_2 newton_iter_5 lambda_learned 1e-1'] = "checkpoints_temp/CG_L2/coco/meta_reg1e-1_iter500_learned/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['coco_meta cg_iter_2 newton_iter_5 lambda_init 1e-1'] = "checkpoints_temp/CG_L2/coco/meta_reg1e-1_iter500_init/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['coco_meta cg_iter_2 newton_iter_5 lambda_none 0']="checkpoints_temp/CG_L2/coco/meta_reg_0/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"


# coco meta warmup and no warmup
# scalar lambda
path['coco_meta cg_iter_2 newton_iter_5 lambda_warmup 1']='checkpoints_temp/CG_L2/coco/meta_reg1_warm/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['coco_meta cg_iter_2 newton_iter_5 lambda_no_warmup 1']='checkpoints_temp/CG_L2/coco/meta_reg1_nowarm/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
# vector lambda
path['coco_meta cg_iter_2 newton_iter_5 lambda_warmup std_1 lr_1e-3']='checkpoints_temp/CG_L2/coco/meta_regvec_std1_lr1e-3_warm/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['coco_meta cg_iter_2 newton_iter_5 lambda_no_warmup std_1 lr_1e-3']='checkpoints_temp/CG_L2/coco/meta_regvec_std1_lr1e-3_nowarm/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['coco_meta cg_iter_2 newton_iter_5 lambda_warmup std_1 lr_5e-3']='checkpoints_temp/CG_L2/coco/meta_regvec_std1_lr5e-3_warm/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['coco_meta cg_iter_2 newton_iter_5 lambda_no_warmup std_1 lr_5e-3']='checkpoints_temp/CG_L2/coco/meta_regvec_std1_lr5e-3_nowarm/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['coco_meta cg_iter_2 newton_iter_5 lambda_warmup std_1 lr_1e-2'] = 'checkpoints_temp/CG_L2/coco/meta_regvec_std1_lr1e-2_warm/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
# vector lambda with values near zero
path['coco_meta cg_iter_2 newton_iter_5 lambda_no_warmup std_5e-1 lr_1e-2']='checkpoints_temp/CG_L2/coco/meta_regvec_std5e-1_lr1e-2_nowarm/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'

path['coco_meta cg_iter_2 newton_iter_5 lambda_warmup std_1e-1 lr_1e-2']='checkpoints_temp/CG_L2/coco/meta_regvec_std1e-1_lr1e-2_warm/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['coco_meta cg_iter_2 newton_iter_5 lambda_no_warmup std_1e-1 lr_1e-2']='checkpoints_temp/CG_L2/coco/meta_regvec_std1e-1_lr1e-2_nowarm/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['coco_meta cg_iter_2 newton_iter_5 lambda_no_warmup_init std_1e-1 lr_1e-2']='checkpoints_temp/CG_L2/coco/meta_regvec_std1e-1_lr1e-2_nowarm_init/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'

path['coco_meta cg_iter_2 newton_iter_5 lambda_warmup std_1e-3 lr_1e-3']='checkpoints_temp/CG_L2/coco/meta_regvec_std1e-3_lr1e-3_warm/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['coco_meta cg_iter_2 newton_iter_5 lambda_no_warmup std_1e-3 lr_1e-3']='checkpoints_temp/CG_L2/coco/meta_regvec_std1e-3_lr1e-3_nowarm/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'

# weight predictor
path['weight_pred lambda std1e-1 lr1e-2']='checkpoints_temp/CG_L2/coco/meta_weight_regvec_std1e-1_lr1e-2_nowarm/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['weight_pred lr1e-2']='checkpoints_temp/CG_L2/coco/meta_weight_lr1e-2_nowarm/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
# feature augmentation
path['CG_L2 cg_iter_2 newton_iter_400 shot_1 aug std_1e-1 drop_3e-1']='checkpoints_temp/CG_L2/coco/cg2_shot1_iter400_aug_std1e-1_drop3e-1/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['CG_L2 cg_iter_2 newton_iter_400 shot_1 aug std_1e-2 drop_2e-1']='checkpoints_temp/CG_L2/coco/cg2_shot1_iter400_aug_std1e-2_drop2e-1/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
# weight predictor retrained
path['weight_pred_simp lr1e-2 newton0']='checkpoints_temp/CG_L2_weight_retrained/coco/meta_simp_weight_lr1e-2_newton0/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['weight_pred_simp lr1e-2 newton5']='checkpoints_temp/CG_L2_weight_retrained/coco/meta_simp_weight_lr1e-2_newton5/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['weight_pred lr1e-2 newton0']='checkpoints_temp/CG_L2_weight_retrained/coco/meta_weight_lr1e-2_newton0_run2/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['weight_pred lr1e-2 newton5']='checkpoints_temp/CG_L2_weight_retrained/coco/meta_weight_lr1e-2_newton5/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'

path['weight_pred lambda lr1e-2 newton5 no_aug']='checkpoints_temp/CG_L2_weight_retrained/coco/meta_weight_regvec_std1e-1_lr1e-2_newton5/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'

path['weight_pred lambda lr1e-2 newton5 aug']='checkpoints_temp/CG_L2_weight_retrained/coco/meta_weight_regvec_std1e-1_lr1e-2_newton5_aug/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['CG_L2 cg_iter_2 shot_1 novel']='checkpoints_temp/CG_L2/coco/cg2_shot1_iter200/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'

path['CG_L2 cg_iter_2 shot_1 randinit']='checkpoints_temp/CG_L2_weight_retrained/coco/meta_weight_randinit/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'

path['feature lr1e-2 newton5']='checkpoints_temp/CG_L2_weight_retrained/coco/meta_feature_lr1e-2_newton5/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'

# coco bAP 
path['CG_L2 cg_iter_2 newton_iter_5 shot_1']="checkpoints_temp/CG_L2_test/coco/cg2_shot1_iter5/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['SGD_L2 feature_extracted num_iter_5 shot_1']="checkpoints_temp/CG_L2_test/coco/sgd2_shot1_iter5/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['CG_L2 cg_iter_2 newton_iter_5 shot_1 score_bias']='checkpoints_temp/CG_L2_test/coco/cg2_shot1_iter200_score_bias/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['CG_L2 cg_iter_2 newton_iter_10 shot_1 two_stage novel']='checkpoints_temp/CG_L2_test/coco/cg2_shot1_iter10_ts_combine/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['CG_L2 cg_iter_2 newton_iter_10 shot_1 two_stage randinit']='checkpoints_temp/CG_L2_test/coco/cg2_shot1_iter10_ts_randinit/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'

# coco fc rigid two-stage detection 
path['SGD_L2 feature_extracted fc shot_1'] = 'checkpoints_temp/CG_L2_test/coco/sgd2_shot1_iter16000_fc/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['SGD_L2 feature_extracted cos shot_1'] = 'checkpoints_temp/SGD_L2_modified/coco/exp_0/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['CG_L2 cg_iter2 newton_iter_200 fc shot_1'] = 'checkpoints_temp/CG_L2_test/coco/cg2_shot1_iter200_fc/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['CG_L2 cg_iter2 newton_iter_200 cos shot_1'] = 'checkpoints_temp/CG_L2/coco/cg2_shot1_iter200/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['CG_L2_RTS cg_iter_2 newton_iter_200 shot_1 randinit']='checkpoints_temp/CG_L2_test/coco/cg2_shot1_iter200_rts_fc_randinit/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['CG_L2_RTS cg_iter_2 newton_iter_200 shot_1 novel'] = 'checkpoints_temp/CG_L2_test/coco/cg2_shot1_iter200_rts_fc_combine/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'

# coco fc RTS filter_3levels+bg
path['RTS SGD_L2 iter_1000 shot_1 randinit']='checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_iter1000_rts_randinit/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
# NOTE no ckpt but loss can be displayed path['Novel SGD_L2 iter_500 shot_1']
path['Novel SGD_L2 iter_1000 shot_1 RTS_data']='checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_novel_iter1000_rtsdata/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json'

path['RTS CG_L2 iter_100 shot_1 randinit test1']='checkpoints_temp/CG_L2_test/coco/cg2_shot1_iter100_rts_fc_randinit_debug_filter_3levels_bg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['RTS CG_L2 iter_60 shot_1 randinit test2']='checkpoints_temp/CG_L2_test/coco/cg2_shot1_iter60_rts_fc_randinit_debug_filter_3levels_bg_test2/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['RTS CG_L2 iter_60 shot_1 randinit test3']='checkpoints_temp/CG_L2_test/coco/cg2_shot1_iter60_rts_fc_randinit_debug_filter_3levels_bg_test3/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['RTS CG_L2 iter_25 shot_1 novel']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_iter25_rts_novel/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['Novel CG_L2 iter_200 shot_1']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter200/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json'

path['RTS CG_L2 iter_200 shot_1 randinit reg_1e-1']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter200_reg1e-1/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json'
path['RTS CG_L2 iter_200 shot_1 randinit reg_1']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter200_reg1/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json'
path['RTS CG_L2 iter_100 shot_1 randinit grad_clip_norm_1']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter100_gradclip_norm1/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json'

# coco novel weights meta learning
# SGD
path['Novel SGD_L2 iter_500 shot_1 seed_0']="checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_novel_iter500_seed0/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json"
path['Novel SGD_L2 iter_500 shot_1 seed_10']="checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_novel_iter500_seed10/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json"
path['Novel SGD_L2 iter_500 shot_1 meta_predweight_tfa']="checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_novel_iter500_metaweight_test/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json"
path['Novel SGD_L2 iter_500 shot_1 meta_predweight_novel']="checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_novel_iter500_metaweight_novel_newton0_iter1000_exp1/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json"
path['Novel SGD_L2 iter_500 shot_1 meta_initweight_novel']="checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_novel_iter500_metainitweight_novel_newton0_iter1000/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json"
path['Novel SGD_L2 iter_500 shot_1 meta_predweight_novel_cg_iter16']="checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_novel_iter300_metaweight_novel_newton0_iter1000_cg_iter16_exp2/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json"
# CG
path['Novel CG_L2 iter_20 shot_1 sgd_weight_iter100_seed0']="checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter20_sgdweightseed0_novel_iter100/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json"
path['Novel CG_L2 iter_20 shot_1 sgd_weight_iter100_seed10']="checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter20_sgdweightseed10_novel_iter100/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json"
path['Novel CG_L2 iter_20 shot_1 meta_predweight_tfa']="checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter20_metaweight_test/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json"
path['Novel CG_L2 iter_20 shot_1 meta_predweight_novel']="checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter20_metaweight_novel_newton0_iter1000/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json"
path['Novel CG_L2 iter_20 shot_1 meta_initweight_novel']="checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter20_metainitweight_novel_newton0_iter1000/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json"

# coco rts hessian and feature augmentation
path['Novel CG_L2 iter_30 shot_1 hessian_init1_factor9e-1']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter30_hessian_init1_factor9e-1/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json'
path['Novel CG_L2 iter_30 shot_1 hessian_init4e-1_factor9e-1']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter30_hessian_baseline/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json'
path['Novel CG_L2 iter_30 shot_5 hessian_init1_factor9e-1']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot5_novel_iter40_hessian_init1_factor9e-1/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json'
path['Novel CG_L2 iter_30 shot_5 hessian_init4e-1_factor9e-1']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot5_novel_iter30_hessian_init4e-1_factor9e-1/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json'
path['Novel CG_L2 iter_30 shot_1 hessian aug_5shot']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter30_hessian_aug5shot/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json'
path['Novel CG_L2 iter_30 shot_1 hessian aug_10shot']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter30_hessian_aug10shot/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/inference/all_res.json'

path['RTS CG_L2 iter_40 shot_1 hessian']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_rts_iter40_hessian/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['RTS CG_L2 iter_40 shot_1 hessian weighted_loss5e-1']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_rts_iter40_hessian_wloss5e-1/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['RTS CG_L2 iter_40 shot_1 hessian feat_aug']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_rts_iter40_hessian_aug5shot_exp3/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['RTS CG_L2 iter_40 shot_2 hessian feat_aug']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot2_rts_iter40_hessian_aug5shot/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['RTS CG_L2 iter_40 shot_3 hessian feat_aug']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot3_rts_iter40_hessian_aug5shot/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['RTS CG_L2 iter_40 shot_5 hessian feat_aug']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot5_rts_iter40_hessian_aug5shot_exp1/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['RTS CG_L2 iter_40 shot_10 hessian']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot10_rts_iter40_hessian/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'
path['RTS CG_L2 iter_40 shot_10 hessian feat_aug']='checkpoints_temp/CG_L2_RTS/coco/cg2_shot10_rts_iter40_hessian_aug5shot/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json'

# COCO HDA new setting
path['HDA_new_setting iter_40 shot_1'] = "checkpoints_HDA_CG_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['HDA_new_setting iter_40 shot_1 feat_aug'] = "checkpoints_HDA_CG_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot_aug/inference/all_res.json"
path['HDA_new_setting iter_40 shot_2'] = "checkpoints_HDA_CG_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot/inference/all_res.json"
path['HDA_new_setting iter_40 shot_2 feat_aug'] = "checkpoints_HDA_CG_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot_aug/inference/all_res.json"
path['HDA_new_setting iter_40 shot_3'] = "checkpoints_HDA_CG_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot/inference/all_res.json"
path['HDA_new_setting iter_40 shot_3 feat_aug'] = "checkpoints_HDA_CG_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot_aug/inference/all_res.json"
path['HDA_new_setting iter_40 shot_5'] = "checkpoints_HDA_CG_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot/inference/all_res.json"
path['HDA_new_setting iter_40 shot_5 feat_aug'] = "checkpoints_HDA_CG_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot_aug/inference/all_res.json"
path['HDA_new_setting iter_40 shot_10'] = "checkpoints_HDA_CG_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot/inference/all_res.json"
path['HDA_new_setting iter_40 shot_10 feat_aug'] = "checkpoints_HDA_CG_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot_aug/inference/all_res.json"

# LVIS TFA+SCG
path['SCG cos_norep newton1 hessian1'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian1/inference/all_res.json"
path['SCG cos_norep newton1 hessian5e-1'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1/inference/all_res.json"
path['SCG cos_norep newton2 hessian1_9e-1'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton2_hessian1_9e-1/inference/all_res.json"
path['SCG cos_norep newton2 hessian5e-1_8e-1'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton2_hessian5e-1_8e-1/inference/all_res.json"
path['SCG cos_norep newton3 hessian1_9e-1'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton3_hessian1_9e-1/inference/all_res.json"
path['SCG cos_norep newton1 hessian5e-1 novel'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_fc_novel_cosine_norepeat_newton1_hessian5e-1/inference/all_res.json"
path['SCG cos_norep newton2 hessian1_9e-1 novel'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_fc_novel_cosine_norepeat_newton2_hessian1_9e-1/inference/all_res.json"
# newly trained small batch TFA+SCG, sinlge newton iter + hessian + batch size 20
path['SCG cos_norep newton1 hessian5e-1 batch20'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1_iter1000/inference/all_res.json"
path['SCG cos_norep newton1 hessian1 batch20'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian1_iter1000/inference/all_res.json"
# COCO TFA+SCG
path['COCO TFA SCG newton1 batch400 iter200']="checkpoints_temp/COCO_TFA_SCG/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot_SCG_test_newton1/inference/all_res.json"
path['COCO TFA SCG newton2 batch400 iter100']="checkpoints_temp/COCO_TFA_SCG/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot_SCG_test_newton2/inference/all_res.json"

# LVIS HDA+SCG
path['HDA SCG dataset_rare novel_r randinit'] = "checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1_iter1000_rare/inference/all_res.json"
path['HDA SCG dataset_all novel_r randinit'] = "checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1_iter1000/inference/all_res.json"
path['HDA SCG dataset_all novel_r combinit'] = "checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1_iter500_combinit/inference/all_res.json"
path['HDA SCG dataset_all novel_rc combinit'] = "checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1_iter500_rc/inference/all_res.json"

# COCO TFA+CG
path['TFA+CG 1-shot']="checkpoints_temp/coco_tfa_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['TFA+CG 2-shot']="checkpoints_temp/coco_tfa_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot/inference/all_res.json"
path['TFA+CG 3-shot']="checkpoints_temp/coco_tfa_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot/inference/all_res.json"
path['TFA+CG 5-shot']="checkpoints_temp/coco_tfa_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot/inference/all_res.json"
path['TFA+CG 10-shot']="checkpoints_temp/coco_tfa_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot/inference/all_res.json"
# COCO TFA+SGD
path['TFA+SGD 1-shot']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['TFA+SGD 2-shot']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot/inference/all_res.json"
path['TFA+SGD 3-shot']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot/inference/all_res.json"
path['TFA+SGD 5-shot']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot/inference/all_res.json"
path['TFA+SGD 10-shot']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot/inference/all_res.json"
# COCO HDA+CG TODO

x_lists = [['TFA+CG 1-shot'],['TFA+CG 2-shot'],['TFA+CG 3-shot'],['TFA+CG 5-shot'],['TFA+CG 10-shot'],
           ['TFA+SGD 1-shot'], ['TFA+SGD 2-shot'], ['TFA+SGD 3-shot'],['TFA+SGD 5-shot'], ['TFA+SGD 10-shot']]

# x_lists = [['TFA+CG 1-shot'], ['TFA+CG 2-shot']]

for x_list in x_lists:
    metrics = {}
    ckpt_keys = {}
    for x in x_list: 
        metrics[x]=[]
        with open(path[x],'r') as f:
            for line in f.readlines():
                metrics[x].append(json.loads(line))
        ckpt_keys[x] = list(metrics[x][0].keys())[2:-1]

    results = {}
    num_iter = []

    # TODO multiply t for corresponding model
    for i,ckpt in enumerate(ckpt_keys[x_list[0]]):
        # if i%2==1:
        num_iter.append(int(ckpt.split('model_')[1].split('.')[0])+1)

    # metric_terms = ["APr", "APc", "APf"]
    metric_terms = ["AP", "bAP", "nAP"]
    for x in x_list:
        results[x]={}
        for metric_term in metric_terms:
            results[x][metric_term] = []
            for i,ckpt in enumerate(ckpt_keys[x]):
                # if i%2==1:
                results[x][metric_term].append(metrics[x][0][ckpt]["bbox"][metric_term])

    # plot
    # y_lims for voc
    # y_lims = {"AP": (34, 43.5), "bAP": (44, 52), "nAP": (0,24)}
    # y_lims for coco
    y_lims = {"AP": (20, 31), "bAP": (25, 40), "nAP": (0, 11)}
    # y_lims = {"AP": (18, 21.5), "bAP": (24, 27.5), "nAP": (0, 3.5)}
    # y_lims for lvis
    # y_lims = {"APr": (0, 19), "APc": (5, 25), "APf": (10, 31)}


    # for metric_term in metric_terms:
    #     plt.figure(figsize=(8, 5))
    #     for x in x_list:
    #         plt.plot(num_iter, results[x][metric_term], '-o', label = x.split(' ')[0]+' '+x.split(' ')[1])
    #     plt.ylim(y_lims[metric_term])
    #     plt.title("{} from checkpoints with period of 1000 (COCO)".format(metric_term))
    #     plt.xlabel("num of iteration")
    #     plt.ylabel(metric_term)
    #     plt.legend()
    #     plt.savefig("checkpoints_temp/Apr07_figs/coco_ckpt_iter_{}_{}.png".format(x_list[0].split('_')[-1], metric_term))

    # colors = {'SGD_L2 feature_extracted': 'green', 
    #           'CG_L2 cg_iter_num_2': 'blue',
    #           'CG_L2 cg_iter_num_3': 'orange'}
    # colors = {'cg_iter_num_2 1e-2': 'bo-',
    #           'cg_iter_num_2 5e-3': 'bo--',
    #           'cg_iter_num_2 1e-3': 'bo:'}
    # colors = {'SGD_L2 feature_extracted': 'go-', 
    #           'SGD_L2 feature_full_batch': 'go--'}
    colors = {'lambda_learned': 'ro-',
            'lambda_init': 'bo-',
            'lambda_none': 'go-',
            'lambda_warmup': 'bo-',
            'lambda_no_warmup': 'mo-',
            'lambda_no_warmup_init': 'mo--',
            'Hessian': 'bo--',
            'Hessian_weighted_loss': 'bo:',
            'Hessian_aug5shot': 'bo-',
            'Hessian_aug10shot': 'bo:',
            'newton5': 'bo-',
            'newton0': 'bo--',
            'novel': 'bo-',
            'randinit': 'bo--',
            'aug': 'bo-',
            'no_aug': 'bo--',
            'TFA+CG (random weights)': 'bo-',
            'TFA+CG': 'bo-',
            'TFA+SGD': 'bo-',
            'HDA+CG': 'bo-',
            'HDA+CG+aug': 'bo-'}

    label_names = ['TFA+CG']
    # label_names = ['TFA+SGD']

    # label_names = ['TFA+CG+meta']
    # label_names = ['HDA+CG']
    # label_names = ['HDA+CG+aug']
    # label_names = ['no_aug', 'aug']

    fig, axs = plt.subplots(3,1, figsize=(6, 10.8))
    # fig, axs = plt.subplots(2,1, figsize=(6, 7.2))

    fontsize_text = "x-large"
    fontsize_num = "large"
    for idx, metric_term in enumerate(metric_terms):
        for j, x in enumerate(x_list):
            label_name = label_names[j]
            axs[idx].plot(num_iter, results[x][metric_term], colors[label_name], label = label_name, markersize = 3)
            # axs[idx].plot(num_iter, results[x][metric_term], color='blue',marker='o', label= metric_term)
            
            # label_name = x.split(' ')[-1]
            # axs[idx].plot(num_iter, results[x][metric_term], colors[label_name], label = label_name, markersize = 3)
            # axs[idx].axvline(x=100, color='r', linestyle=':')
        axs[idx].set_xticks(num_iter)
        axs[idx].set_ylim(y_lims[metric_term])
        axs[idx].set_ylabel(metric_term,fontsize=fontsize_text)
        # axs[idx].legend()
    # info = x_list[0].split(' ')

    axs[0].set_title('{}'.format(x_list[0]),fontsize=fontsize_text)
    # axs[0].set_title(x_list[0].split('hessian')[0]+'hessian_reg feat_aug')
    # axs[0].set_title('Metrics on predicted weight (linear)')
    # axs[0].set_title('Metrics on predicted weight (linear+relu+linear)')
    # axs[0].set_title('Metrics on predicted weight and lambda (jointly learned)')
    # axs[0].set_title('Metrics on COCO 1-shot dataset')
    # axs[0].set_title('Metrics on projected feature (linear)')

    axs[-1].set_xlabel('number of iterations',fontsize=fontsize_text)
    plt.tight_layout()
    # plt.savefig("checkpoints_temp/All_figs/May26_figs/coco_meta_ckpt_weight_pred_simp.png")
    plt.savefig("checkpoints_temp/figs/Nov13_figs_convergence/{}.png".format(x_list[0]))
    # plt.savefig("checkpoints_temp/All_figs/May26_figs/coco_meta_ckpt_weight_pred_lambda.png")
    # plt.savefig("checkpoints_temp/All_figs/May26_figs/coco_meta_shot1_iter200.png")
    # plt.savefig("checkpoints_temp/All_figs/May26_figs/coco_meta_feat_proj.png")
