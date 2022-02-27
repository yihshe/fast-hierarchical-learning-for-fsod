import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from IPython import embed
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import json
from fsdet.data.builtin_meta import PASCAL_VOC_ALL_CATEGORIES, PASCAL_VOC_BASE_CATEGORIES, PASCAL_VOC_NOVEL_CATEGORIES


id_freq_map = {}
for id in range(20):
    if id<15:
        id_freq_map[id] = "base"
    else:
        id_freq_map[id] = "novel"
id_freq_map[20] = 'bg'

paths = dict({})
# paths["voc hda_cg 5shot"]="checkpoints_temp/voc_infer/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_iter100/inference/voc_instances_results.pth"
# paths["voc tfa_sgd_reported 5shot"]="checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_infer/inference/voc_instances_results.pth"

# paths["voc hda_cg 1shot"]="checkpoints_temp/voc_infer/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel_iter100/inference/voc_instances_results.pth"
# paths["voc hda_cg 10shot"]="checkpoints_temp/voc_infer/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_10shot_randnovel_iter100/inference/voc_instances_results.pth"
# paths["voc tfa_sgd_reported 1shot"]="checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_1shot_randnovel_infer/inference/voc_instances_results.pth"
# paths["voc tfa_sgd_reported 10shot"]="checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_10shot_randnovel_infer/inference/voc_instances_results.pth"
# paths["voc hda_newly_trained base_thresh_5e-1"]="checkpoints_temp/voc_infer/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_thresh_5e-1/inference/voc_instances_results.pth"
paths["voc hda_newly_trained base5e-1 filter5e-1"]="checkpoints_temp/voc_infer/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_filterthresh_5e-1_basethresh_5e-1/inference/voc_instances_results.pth"
paths["voc hda_newly_trained base5e-2 filter5e-1"]="checkpoints_temp/voc_infer/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_filterthresh_5e-1_basethresh_5e-2/inference/voc_instances_results.pth"

for key, path in paths.items():
    objs = torch.load(path)
    plot_params = {
                'figsize': (7,5),
                'annot': True,
                'linewidth': 0.3,
                'title': 'voc conf_mat {}'.format(key),
            }
    
    dts = []
    gts = []
    for obj in objs:
        dts.append(id_freq_map[int(obj["category_id"])])
        gts.append(id_freq_map[int(obj["category_id_gt"])])
    
    labels = ['base', 'novel']

    mat = confusion_matrix(gts, dts, labels=labels)
    # recall = mat.astype(np.float)/mat.sum(axis=1)[:, np.newaxis]
    # precision = mat.astype(np.float)/mat.sum(axis=0)[np.newaxis, :]
    
    plot = pd.DataFrame(mat, index=labels, columns=labels)
    # plot = pd.DataFrame(recall, index=labels, columns=labels)
    # plot = pd.DataFrame(precision, index=labels, columns=labels)
    plt.figure(figsize = plot_params['figsize'])
    sn.heatmap(plot, annot=plot_params['annot'], linewidths=plot_params['linewidth'], cmap="YlGnBu")
    plt.xlabel('Pred', fontsize = 13)
    plt.ylabel('True', fontsize = 13)
    plt.title(plot_params['title'], fontsize = 14)
    plt.savefig('checkpoints_temp/figs/Nov05_figs/{}.png'.format(plot_params['title']))

    # embed()

    
