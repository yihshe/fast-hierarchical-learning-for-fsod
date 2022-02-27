# This script is used to plot metrics for saved checkpoints in real FsDet task

import json
from typing_extensions import runtime
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

path = {}
speed = {
    'TFA+CG 1-Shot': 0.0378,
    'TFA+CG 2-Shot': 0.0687,
    'TFA+CG 3-Shot': 0.1024,
    'TFA+CG 5-Shot': 0.1683,
    'TFA+CG 10-Shot': 0.3339,
    'TFA+SGD 1-Shot': 0.0067,
    'TFA+SGD 2-Shot': 0.0067,
    'TFA+SGD 3-Shot': 0.0068,
    'TFA+SGD 5-Shot': 0.0070,
    'TFA+SGD 10-Shot': 0.0071,
    'HDA+CG+aug 1-Shot': 0.0211,
    'HDA+CG+aug 2-Shot': 0.0359,
    'HDA+CG+aug 3-Shot': 0.0513,
    'HDA+CG+aug 5-Shot': 0.0822,
    'HDA+CG+aug 10-Shot': 0.1589,

}
# COCO TFA+CG
path['TFA+CG 1-Shot']="checkpoints_temp/coco_tfa_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['TFA+CG 2-Shot']="checkpoints_temp/coco_tfa_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot/inference/all_res.json"
path['TFA+CG 3-shot']="checkpoints_temp/coco_tfa_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot/inference/all_res.json"
path['TFA+CG 5-Shot']="checkpoints_temp/coco_tfa_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot/inference/all_res.json"
path['TFA+CG 10-Shot']="checkpoints_temp/coco_tfa_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot/inference/all_res.json"
# COCO TFA+SGD
path['TFA+SGD 1-Shot']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['TFA+SGD 2-Shot']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot/inference/all_res.json"
path['TFA+SGD 3-Shot']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot/inference/all_res.json"
path['TFA+SGD 5-Shot']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot/inference/all_res.json"
path['TFA+SGD 10-Shot']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot/inference/all_res.json"

path['TFA+SGD 1-Shot plus']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot_plus/inference/all_res.json"
path['TFA+SGD 2-Shot plus']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot_plus/inference/all_res.json"
path['TFA+SGD 3-Shot plus']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot_plus/inference/all_res.json"
path['TFA+SGD 5-Shot plus']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot_plus/inference/all_res.json"
path['TFA+SGD 10-Shot plus']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot_plus/inference/all_res.json"

path['TFA+SGD 1-Shot plus2']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot_plus2/inference/all_res.json"
path['TFA+SGD 2-Shot plus2']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot_plus2/inference/all_res.json"
path['TFA+SGD 3-Shot plus2']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot_plus2/inference/all_res.json"
path['TFA+SGD 5-Shot plus2']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot_plus2/inference/all_res.json"
path['TFA+SGD 10-Shot plus2']="checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot_plus2/inference/all_res.json"

# COCO HDA+CG TODO
path['HDA+CG+aug 1-Shot']="checkpoints_temp/coco_hda_cg_aug/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/all_res.json"
path['HDA+CG+aug 2-Shot']="checkpoints_temp/coco_hda_cg_aug/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot/inference/all_res.json"
path['HDA+CG+aug 3-Shot']="checkpoints_temp/coco_hda_cg_aug/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot/inference/all_res.json"
path['HDA+CG+aug 5-Shot']="checkpoints_temp/coco_hda_cg_aug/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot/inference/all_res.json"
path['HDA+CG+aug 10-Shot']="checkpoints_temp/coco_hda_cg_aug/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot/inference/all_res.json"
# x_lists = [['TFA+CG 1-shot'],['TFA+CG 2-shot'],['TFA+CG 3-shot'],['TFA+CG 5-shot'],['TFA+CG 10-shot'],
#            ['TFA+SGD 1-shot'], ['TFA+SGD 2-shot'], ['TFA+SGD 3-shot'],['TFA+SGD 5-shot'], ['TFA+SGD 10-shot']]

x_lists = [
    # ['TFA+SGD 1-Shot', 'HDA+CG+aug 1-Shot'],
    # ['TFA+SGD 2-Shot', 'HDA+CG+aug 2-Shot'],
    ['TFA+SGD 3-Shot', 'HDA+CG+aug 3-Shot'],
    # ['TFA+SGD 5-Shot', 'HDA+CG+aug 5-Shot'],
    # ['TFA+SGD 10-Shot', 'HDA+CG+aug 10-Shot']
    ]

metric_terms = ["AP", "bAP", "nAP"]

for x_list in x_lists:
    metrics = {}
    ckpt_keys = {}
    results = {}
    running_time = {}
    for x in x_list: 
        metrics[x]=[]
        with open(path[x],'r') as f:
            for line in f.readlines():
                metrics[x].append(json.loads(line))
        ckpt_keys[x] = list(metrics[x][0].keys())[2:-1]
        
        running_time[x] = []
        for i, ckpt in enumerate(ckpt_keys[x]):
            running_time[x].append((int(ckpt.split('model_')[1].split('.')[0])+1)*speed[x])

        results[x]={}
        for metric_term in metric_terms:
            results[x][metric_term] = []
            for i,ckpt in enumerate(ckpt_keys[x]):
                # if i%2==1:
                results[x][metric_term].append(metrics[x][0][ckpt]["bbox"][metric_term])

        if "TFA+SGD" in x:
            x2 = "{} plus".format(x)
            metrics2 = []
            running_time2 = []
            results2 = {}
            with open(path[x2],'r') as f:
                for line in f.readlines():
                    metrics2.append(json.loads(line))
            ckpt_keys2 = list(metrics2[0].keys())[2:-1]

            for i, ckpt in enumerate(ckpt_keys2):
                running_time2.append((int(ckpt.split('model_')[1].split('.')[0])+1)*speed[x])

            for metric_term in metric_terms:
                results2[metric_term] = []
                for i,ckpt in enumerate(ckpt_keys2):
                    # if i%2==1:
                    results2[metric_term].append(metrics2[0][ckpt]["bbox"][metric_term])
            
            # running_time[x] = running_time2[:-1]+running_time[x]
            running_time[x] = running_time2+running_time[x][1:]
            for metric_term in metric_terms:
                # results[x][metric_term]=results2[metric_term][:-1]+results[x][metric_term]
                results[x][metric_term]=results2[metric_term]+results[x][metric_term][1:]

            x3 = "{} plus2".format(x)
            metrics3 = []
            running_time3 = []
            results3 = {}
            with open(path[x3],'r') as f:
                for line in f.readlines():
                    metrics3.append(json.loads(line))
            ckpt_keys3 = list(metrics3[0].keys())[2:-1]

            for i, ckpt in enumerate(ckpt_keys3):
                running_time3.append((int(ckpt.split('model_')[1].split('.')[0])+1)*speed[x])

            for metric_term in metric_terms:
                results3[metric_term] = []
                for i,ckpt in enumerate(ckpt_keys3):
                    # if i%2==1:
                    results3[metric_term].append(metrics3[0][ckpt]["bbox"][metric_term])
            
            running_time[x] = running_time3+running_time[x]
            for metric_term in metric_terms:
                results[x][metric_term]=results3[metric_term]+results[x][metric_term]

    # results = {}
    # num_iter = []

    # TODO multiply t for corresponding model
    # for i,ckpt in enumerate(ckpt_keys[x_list[0]]):
    #     # if i%2==1:
    #     num_iter.append(int(ckpt.split('model_')[1].split('.')[0])+1)

    # metric_terms = ["APr", "APc", "APf"]
    # metric_terms = ["AP", "bAP", "nAP"]
    # for x in x_list:
    #     results[x]={}
    #     for metric_term in metric_terms:
    #         results[x][metric_term] = []
    #         for i,ckpt in enumerate(ckpt_keys[x]):
    #             # if i%2==1:
    #             results[x][metric_term].append(metrics[x][0][ckpt]["bbox"][metric_term])

    # plot
    # y_lims for voc
    # y_lims = {"AP": (34, 43.5), "bAP": (44, 52), "nAP": (0,24)}
    # y_lims for coco
    # y_lims = {"AP": (20, 31), "bAP": (25, 40), "nAP": (0, 11)}
    y_lims = {"AP": (24, 35), "bAP": (32, 40), "nAP": (-0.5, 10)}
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

    colors = {'TFA+CG': 'green', 
              'TFA+SGD': 'blue',
              'HDA+CG+aug': 'orange'}
    # colors = {'cg_iter_num_2 1e-2': 'bo-',
    #           'cg_iter_num_2 5e-3': 'bo--',
    #           'cg_iter_num_2 1e-3': 'bo:'}
    # colors = {'SGD_L2 feature_extracted': 'go-', 
    #           'SGD_L2 feature_full_batch': 'go--'}
    # colors = {'lambda_learned': 'ro-',
            # 'lambda_init': 'bo-',
            # 'lambda_none': 'go-',
            # 'lambda_warmup': 'bo-',
            # 'lambda_no_warmup': 'mo-',
            # 'lambda_no_warmup_init': 'mo--',
            # 'Hessian': 'bo--',
            # 'Hessian_weighted_loss': 'bo:',
            # 'Hessian_aug5shot': 'bo-',
            # 'Hessian_aug10shot': 'bo:',
            # 'newton5': 'bo-',
            # 'newton0': 'bo--',
            # 'novel': 'bo-',
            # 'randinit': 'bo--',
            # 'aug': 'bo-',
            # 'no_aug': 'bo--',
            # 'TFA+CG (random weights)': 'bo-',
            # 'TFA+CG': 'bo-',
            # 'TFA+SGD': 'bo-',
            # 'HDA+CG': 'bo-',
            # 'HDA+CG+aug': 'bo-'}

    name_map = {"TFA+SGD": "TFA*", "HDA+CG+aug": "HDA"}
    x_axs_map = {
        "1-Shot": {"xticks": [0.1,1,10,100], "xticklabels": [0.1,1,10,100]},
        "2-Shot": {"xticks": [1,10,100], "xticklabels": [1,10,100]},
        "3-Shot": {"xticks": [1,10,100], "xticklabels": [1,10,100]},
        "5-Shot": {"xticks": [1,10,100], "xticklabels": [1,10,100]},
        "10-Shot": {"xticks": [1,10,100,1000], "xticklabels": [1,10,100,1000]},
        }
    y_axs_map = {
        "AP": {"yticks": range(24,35,2), "yticklabels": range(24,35,2)},
        "bAP": {"yticks": range(32,41,2), "yticklabels": range(32,41,2)},
        "nAP": {"yticks": range(0,11,2), "yticklabels": range(0,11,2)},
    }
    
    # fig, axs = plt.subplots(3,1, figsize=(6, 10.8))
    fig, axs = plt.subplots(1,3, figsize=(13, 3))
    fontsize_text = "xx-large"
    fontsize_legend = "large"
    fontsize_num = "x-large"
    for idx, metric_term in enumerate(metric_terms):
        for j, x in enumerate(x_list):
            # label_name = label_names[j]
            label_name = x.split(' ')[0]
            axs[idx].plot(running_time[x], results[x][metric_term], color = colors[label_name], marker = 'o', label = name_map[label_name], markersize = 3)
            # axs[idx].plot(num_iter, results[x][metric_term], color='blue',marker='o', label= metric_term)
            
            # label_name = x.split(' ')[-1]
            # axs[idx].plot(num_iter, results[x][metric_term], colors[label_name], label = label_name, markersize = 3)
            # axs[idx].axvline(x=100, color='r', linestyle=':')
        
        # axs[idx].set_xticks(num_iter)
        shot = x_list[0].split(" ")[1]
        axs[idx].set_xscale('log', base=10)
        axs[idx].set_xticks(x_axs_map[shot]["xticks"])
        axs[idx].set_xticklabels(x_axs_map[shot]["xticklabels"],fontsize=fontsize_num)
        axs[idx].set_ylim(y_lims[metric_term])
        axs[idx].set_yticks(y_axs_map[metric_term]["yticks"])
        axs[idx].set_yticklabels(y_axs_map[metric_term]["yticklabels"],fontsize=fontsize_num)
        axs[idx].set_ylabel(metric_term,fontsize=fontsize_text)
        axs[idx].legend(prop={'size': fontsize_legend})
    # info = x_list[0].split(' ')

    axs[1].set_title('{}'.format(x_list[0].split(' ')[1]),fontsize=fontsize_text)
    # axs[0].set_title(x_list[0].split('hessian')[0]+'hessian_reg feat_aug')
    # axs[0].set_title('Metrics on predicted weight (linear)')
    # axs[0].set_title('Metrics on predicted weight (linear+relu+linear)')
    # axs[0].set_title('Metrics on predicted weight and lambda (jointly learned)')
    # axs[0].set_title('Metrics on COCO 1-shot dataset')
    # axs[0].set_title('Metrics on projected feature (linear)')

    axs[1].set_xlabel('Time in seconds (log scale)',fontsize=fontsize_text)
    plt.tight_layout()
    # plt.savefig("checkpoints_temp/All_figs/May26_figs/coco_meta_ckpt_weight_pred_simp.png")
    plt.savefig("checkpoints_temp/figs/Nov15_convergence_log/{}_{}_{}_horizontal.png".format(x_list[0].split(' ')[0], x_list[1].split(' ')[0], x_list[0].split(' ')[1],))
    # plt.savefig("checkpoints_temp/All_figs/May26_figs/coco_meta_ckpt_weight_pred_lambda.png")
    # plt.savefig("checkpoints_temp/All_figs/May26_figs/coco_meta_shot1_iter200.png")
    # plt.savefig("checkpoints_temp/All_figs/May26_figs/coco_meta_feat_proj.png")
