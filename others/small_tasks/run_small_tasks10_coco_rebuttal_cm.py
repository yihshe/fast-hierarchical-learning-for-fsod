import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from IPython import embed
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import json
from fsdet.data.builtin_meta import _get_builtin_metadata

metadata = _get_builtin_metadata("coco_fewshot")

def init_idmaps(metadata):
    idmap_global = metadata['thing_dataset_id_to_contiguous_id']
    idmap_global_reversed = {v: k for k, v in idmap_global.items()}

    idmap_base = metadata['base_dataset_id_to_contiguous_id']
    idmap_base_reversed = {v: k for k, v in idmap_base.items()}
    base_class_ids_global = [idmap_global[k] for k in idmap_base.keys()]

    idmap_novel = metadata['novel_dataset_id_to_contiguous_id']
    idmap_novel_reversed = {v: k for k, v in idmap_novel.items()}
    novel_class_ids_global = [idmap_global[k] for k in idmap_novel.keys()]

    return {
        'idmap_global': idmap_global,
        'idmap_global_reversed': idmap_global_reversed,
        'idmap_base': idmap_base,
        'idmap_base_reversed': idmap_base_reversed,
        'base_class_ids_global': base_class_ids_global,
        'idmap_novel': idmap_novel,
        'idmap_novel_reversed': idmap_novel_reversed,
        'novel_class_ids_global': novel_class_ids_global,
    }

idmap = init_idmaps(metadata)

# the following codes were used for distribution analysis
pred_vs_true = torch.load("checkpoints_temp/coco_hda_cg_aug/rebuttal_10shot/pred_vs_true_10shot_standard_setting.pth")
pred = pred_vs_true['pred']
true = pred_vs_true['true']
true_ids = idmap["novel_class_ids_global"]
mat = torch.zeros(len(true_ids), 61)
for i, true_id in enumerate(true_ids):
    mat[i] = torch.bincount(pred[true==true_id], minlength=61)
mat_norm = torch.tensor(np.array(mat).astype(np.float)/np.array(mat).sum(axis=1)[:, np.newaxis])
mat_row_cats = metadata["novel_classes"]
mat_col_cats = metadata["base_classes"]+["bg"]
ret = {"mat": mat, "mat_norm": mat_norm, "row_cats": mat_row_cats, "col_cats": mat_col_cats}
torch.save(ret, "rebuttal_pred_class_standard_setting_10shot.pth")


# NOTE the following codes were used to generate the confusion matrix plot
# id2cat = {}
# for id in range(80):
#     if id in idmap["base_class_ids_global"]:
#         id2cat[id] = "base"
#     elif id in idmap["novel_class_ids_global"]:
#         id2cat[id] = "novel"
# id2cat[80] = "bg"

# paths = {}
# paths["10shot"] = "checkpoints_temp/coco_hda_cg_aug_infer/faster_rcnn/faster_rcnn_R_101_FPN_ft_fc_all_10shot/inference/instances_predictions.pth"

# objs = torch.load(paths["10shot"])
# dts = []
# gts = []
# for obj in objs:
#     for ins in obj["instances"]:
#         dts.append(id2cat[int(ins["category_id"])])
#         gts.append(id2cat[int(ins["category_id_gt"])])

# labels = ["base", "novel"]
# mat = confusion_matrix(gts, dts, labels=labels)

# recall = mat.astype(np.float)/mat.sum(axis=1)[:, np.newaxis]
# precision = mat.astype(np.float)/mat.sum(axis=0)[np.newaxis, :]
    
# plot_params = {
#                 'figsize': (7,5),
#                 'annot': True,
#                 'linewidth': 0.3,
#                 'title': 'HDA CG AUG precision',
#             }

# plot = pd.DataFrame(precision, index=labels, columns=labels)
# plt.figure(figsize=plot_params["figsize"])
# sn.heatmap(plot, annot=plot_params['annot'], linewidths=plot_params['linewidth'], cmap="YlGnBu")
# plt.xlabel('Pred', fontsize = 13)
# plt.ylabel('True', fontsize = 13)
# plt.title(plot_params['title'], fontsize = 14)
# plt.savefig('rebuttal_{}.png'.format(plot_params['title']))
