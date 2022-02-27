import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from IPython import embed
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import json
from fsdet.data.lvis_v0_5_categories import LVIS_CATEGORIES, LVIS_CATEGORIES_BASE, LVIS_CATEGORIES_NOVEL

# freq_id_map = {}
# for cat in LVIS_CATEGORIES:
#     if cat['frequency'] not in freq_id_map.keys():
#         freq_id_map[cat['frequency']] = list([cat['id']])
#     else:
#         freq_id_map[cat['frequency']].append(cat['id'])
# embed()
id_freq_map = {}
for cat in LVIS_CATEGORIES:
    id_freq_map[cat['id']] = cat['frequency']
id_freq_map[int(1231)] = 'bg'

paths = dict({})
# paths["tfa pretrain_base"] = "checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_norepeat_cos_infer/inference/lvis_instances_results.json"
# paths["tfa init comb"] = "checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_infer_comb_init/inference/lvis_instances_results.json"
# paths["tfa final combinit"] = "checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_infer_final/inference/lvis_instances_results.json"
# paths["hda init comb"] = "checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_infer_comb_init/inference/lvis_instances_results.json"
# paths["hda final randinit_rare_r"] = "checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_infer_final_rare/inference/lvis_instances_results.json"

# paths["hda final exp1_randinit_all_r"]="checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_infer_final_exp1/inference/lvis_instances_results.json"
# paths["hda final exp2_combinit_all_rc"]="checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_infer_final_exp2_rc/inference/lvis_instances_results.json"
paths["hda final exp1_combinit_all_r"]="checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_infer_final_exp1_combinit/inference/lvis_instances_results.json"

for key, path in paths.items():
    with open(path) as f:
        objs = json.load(f)

    plot_params = {
                'figsize': (7,5),
                'annot': True,
                'linewidth': 0.3,
                'title': 'lvis conf_mat {}'.format(key),
            }

    dts = []
    gts = []
    for obj in objs:
        dts.append(id_freq_map[int(obj["category_id"])])
        gts.append(id_freq_map[int(obj["category_id_gt"])])
    
    labels = ['f', 'c', 'r']
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
    plt.savefig('checkpoints_temp/figs/Oct29_figs/{}.png'.format(plot_params['title']))

    # embed()

    
