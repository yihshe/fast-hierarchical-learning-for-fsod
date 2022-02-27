# This script is used to plot the losses and lambda in meta training

# Plot metrics for fine tuning
import json
import matplotlib.pyplot as plt
import numpy as np
from IPython.terminal.embed import embed

metrics={}
path = {}

# Meta
# query loss
path['coco_meta init_lambda_1 iter_500 query_loss no_warmup'] = 'checkpoints_temp/CG_L2_meta/coco_meta_reg1_iter500/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['coco_meta init_lambda_1 iter_500 query_loss warmup']='checkpoints_temp/CG_L2_meta/coco_meta_reg1_iter500_warm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'

path['coco_meta init_lambda_5e-1 iter_500 query_loss'] = 'checkpoints_temp/CG_L2_meta/coco_meta_reg5e-1_iter500/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['coco_meta init_lambda_1e-1 iter_500 query_loss'] = 'checkpoints_temp/CG_L2_meta/coco_meta_reg1e-1_iter500/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'

path['lambda_vec iter_1000 std1 lr_1e-3 query_loss warm']='checkpoints_temp/CG_L2_meta/coco_meta_reg_randvec_iter1000_lr1e-3_warm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['lambda_vec iter_1000 std1 lr_1e-3 query_loss no_warm']='checkpoints_temp/CG_L2_meta/coco_meta_reg_randvec_iter1000_lr1e-3_nowarm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['lambda_vec iter_1000 std1 lr_5e-3 query_loss warm']='checkpoints_temp/CG_L2_meta/coco_meta_reg_randvec_iter1000_lr5e-3_warm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['lambda_vec iter_1000 std1 lr_5e-3 query_loss no_warm']='checkpoints_temp/CG_L2_meta/coco_meta_reg_randvec_iter1000_lr5e-3_nowarm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['lambda_vec iter_1000 std1 lr_1e-2 query_loss warm'] = 'checkpoints_temp/CG_L2_meta/coco_meta_reg_randvec_iter1000_lr1e-2_warm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'

# coco meta vector lambda std near 0

path['lambda_vec iter_1000 std5e-1 lr_1e-2 query_loss no_warm']='checkpoints_temp/CG_L2_meta/coco_meta_reg_randvec_std5e-1_iter1000_lr1e-2_nowarm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'

path['lambda_vec iter_1000 std1e-1 lr_1e-2 query_loss warm']='checkpoints_temp/CG_L2_meta/coco_meta_reg_randvec_std1e-1_iter1000_lr1e-2_warm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['lambda_vec iter_1000 std1e-1 lr_1e-2 query_loss no_warm']='checkpoints_temp/CG_L2_meta/coco_meta_reg_randvec_std1e-1_iter1000_lr1e-2_nowarm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['lambda_vec iter_1000 std1e-1 lr_1e-3 query_loss warm']='checkpoints_temp/CG_L2_meta/coco_meta_reg_randvec_std1e-1_iter1000_lr1e-3_warm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['lambda_vec iter_1000 std1e-1 lr_1e-3 query_loss no_warm']='checkpoints_temp/CG_L2_meta/coco_meta_reg_randvec_std1e-1_iter1000_lr1e-3_nowarm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'

path['lambda_vec iter_1000 std1e-3 lr_1e-2 query_loss warm']='checkpoints_temp/CG_L2_meta/coco_meta_reg_randvec_std1e-3_iter1000_lr1e-2_warm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['lambda_vec iter_1000 std1e-3 lr_1e-2 query_loss no_warm']='checkpoints_temp/CG_L2_meta/coco_meta_reg_randvec_std1e-3_iter1000_lr1e-2_nowarm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['lambda_vec iter_1000 std1e-3 lr_1e-3 query_loss warm']='checkpoints_temp/CG_L2_meta/coco_meta_reg_randvec_std1e-3_iter1000_lr1e-3_warm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['lambda_vec iter_1000 std1e-3 lr_1e-3 query_loss no_warm']='checkpoints_temp/CG_L2_meta/coco_meta_reg_randvec_std1e-3_iter1000_lr1e-3_nowarm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
# scalar lambda
path['coco_meta init_lambda_1 iter_500 lambda no_warmup'] = 'checkpoints_temp/CG_L2_meta/coco_meta_reg1_iter500/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/lambda.json'
path['coco_meta init_lambda_1 iter_500 lambda warmup'] = 'checkpoints_temp/CG_L2_meta/coco_meta_reg1_iter500_warm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/lambda.json'
path['coco_meta init_lambda_5e-1 iter_500 lambda'] = 'checkpoints_temp/CG_L2_meta/coco_meta_reg5e-1_iter500/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/lambda.json'
path['coco_meta init_lambda_1e-1 iter_500 lambda'] = 'checkpoints_temp/CG_L2_meta/coco_meta_reg1e-1_iter500/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/lambda.json'

# coco meta weight predictor (and vector lambda std near 0)
path['weight_pred lambda_vec_std1e-1 iter_1000 lr_1e-2 query_loss'] = 'checkpoints_temp/CG_L2_meta/coco_meta_regvec_std1e-1_weight_iter1000_lr1e-2_nowarm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['weight_pred iter_1000 lr_1e-2 query_loss'] = 'checkpoints_temp/CG_L2_meta/coco_meta_weight_iter1000_lr1e-2_nowarm/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['weight_pred init iter_1000 lr_1e-2 query_loss']='checkpoints_temp/CG_L2_meta/coco_meta_weight_iter1000_lr1e-2_init/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
# weight predictor retrained
path['weight_pred newton0 iter_1000 lr_1e-2 query_loss']='checkpoints_temp/CG_L2_meta_weight_retrained/coco_meta_weight_iter1000_lr1e-2_newton0/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['weight_pred newton5 iter_1000 lr_1e-2 query_loss']='checkpoints_temp/CG_L2_meta_weight_retrained/coco_meta_weight_iter1000_lr1e-2_newton5/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['weight_pred lambda_vec iter_1000 lr_1e-2 query_loss']='checkpoints_temp/CG_L2_meta_weight_retrained/coco_meta_weight_regvec_std1e-1_iter1000_lr1e-2/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['weight_pred_simp newton0 iter_1000 lr_1e-2 query_loss']='checkpoints_temp/CG_L2_meta_weight_retrained/coco_meta_simp_weight_iter1000_lr1e-2_newton0/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['weight_pred_simp newton5 iter_1000 lr_1e-2 query_loss']='checkpoints_temp/CG_L2_meta_weight_retrained/coco_meta_simp_weight_iter1000_lr1e-2_newton5/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'
path['feature_proj iter_1000 lr_1e-2 query_loss']='checkpoints_temp/CG_L2_meta_weight_retrained/coco_meta_feature_iter1000_lr1e-2/faster_rcnn_R_101_FPN_ft_all_1shot/training_metrics/query_loss.json'

# weight predictor rts novel
path['novel weight_pred newton0 iter_1000 lr_1e-2 query_loss']='checkpoints_temp/CG_L2_RTS_meta/coco_meta_weight_iter500_lr1e-2_newton0/faster_rcnn_R_101_FPN_ft_novel_1shot/training_metrics/query_loss.json'
path['novel weight_pred newton0 iter_1000 lr_1e-2 shot_5 query_loss']='checkpoints_temp/CG_L2_RTS_meta/coco_meta_weight_iter1000_lr1e-2_newton0_shot5/faster_rcnn_R_101_FPN_ft_novel_1shot/training_metrics/query_loss.json'
path['novel weight_init newton0 iter_1000 lr_1e-2 query_loss']='checkpoints_temp/CG_L2_RTS_meta/coco_meta_initweight_iter1000_lr1e-2_newton0/faster_rcnn_R_101_FPN_ft_novel_1shot/training_metrics/query_loss.json'

# LVIS TFA+SCG loss
path['loss: SCG cos_norep newton1 hessian1'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian1/metrics.json"
path['loss: SCG cos_norep newton1 hessian5e-1'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1/metrics.json"
path['loss: SCG cos_norep newton2 hessian1_9e-1'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton2_hessian1_9e-1/metrics.json"
path['loss: SCG cos_norep newton2 hessian5e-1_8e-1'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton2_hessian5e-1_8e-1/metrics.json"
path['loss: SCG cos_norep newton3 hessian1_9e-1'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton3_hessian1_9e-1/metrics.json"
path['loss: SCG cos_norep newton1 hessian5e-1 novel'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_fc_novel_cosine_norepeat_newton1_hessian5e-1/metrics.json"
path['loss: SCG cos_norep newton2 hessian1_9e-1 novel'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_fc_novel_cosine_norepeat_newton2_hessian1_9e-1/metrics.json"
path['loss: SCG cos_norep newton1 hessian5e-1 batch20'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1_iter1000/metrics.json"
path['loss: SCG cos_norep newton1 hessian1 batch20'] = "checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian1_iter1000/metrics.json"
# LVIS TFA+SGD loss
path['loss: SGD cos_norep combined_weight'] = "checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat/metrics.json"
path['loss: SGD cos_norep randinit_weight'] = "checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_randinit_all_norepeat/metrics.json"

paths = [
        # 'loss: SCG cos_norep newton1 hessian1',
        # 'loss: SCG cos_norep newton2 hessian1_9e-1',
        # 'loss: SCG cos_norep newton2 hessian5e-1_8e-1',
        # 'loss: SCG cos_norep newton1 hessian5e-1 batch20',
        # 'loss: SCG cos_norep newton1 hessian1 batch20',
        'loss: SGD cos_norep combined_weight',
        'loss: SGD cos_norep randinit_weight'
        ]

def plot_result(x):
    metrics[x]=[]
    with open(path[x],'r') as f:
        for line in f.readlines():
            metrics[x].append(json.loads(line))

    losses ={}
    losses[x]={}
    loss_names = ['total_loss', 'loss_cls', 'loss_box_reg'] 
    # loss_names = ['lambda', 'grad']

    # TODO loss wieght is not used, nor the -1 should be removed
    for loss_name in loss_names:
        losses[x][loss_name] = [i[loss_name] for i in metrics[x][:-3]]

    xAxis_ticks={}
    xAxis_ticks[x] = [i['iteration'] for i in metrics[x][:-3]]
    y_lims = {'total_loss': [0.1, 0.45], 'loss_cls': [0.1, 0.3], 'loss_box_reg': [0, 0.08], 
            'loss_weight': [2, 18.5], 'lambda': [-0.1, 1.1], 'grad': [-3, 6.5]}
    # 3.6
    fig, axs = plt.subplots(3,1, figsize=(6,10.8)) 
    for idx, loss_name in enumerate(loss_names):
        axs[idx].plot(xAxis_ticks[x], losses[x][loss_name], label=loss_name)
        axs[idx].set_ylim(y_lims[loss_name])
        axs[idx].legend(loc='upper right', fontsize=12)
    axs[0].set_title('{}'.format(x))
    axs[-1].set_xlabel('num of iteration', fontsize=12)

    plt.tight_layout()
    plt.savefig('checkpoints_temp/All_figs/Sep29_figs/{}.png'. format(x))

for x in paths:
    plot_result(x)
