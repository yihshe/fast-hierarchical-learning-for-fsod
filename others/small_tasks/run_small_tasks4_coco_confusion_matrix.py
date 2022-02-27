import numpy as np
from numpy.testing._private.utils import suppress_warnings
import torch
from sklearn.metrics import confusion_matrix
from IPython import embed
from fsdet.data import HDAMetaInfo
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def init_idmaps(metadata, hda_meta_info: HDAMetaInfo):
    idmap_hier1_global = hda_meta_info.get_meta_hda_base()['base_dataset_id_to_contiguous_id']
    idmap_hier1_global_reversed = {v: k for k, v in idmap_hier1_global.items()}

    idmap_hier2_global = metadata['thing_dataset_id_to_contiguous_id']
    idmap_hier2_global_reversed = {v: k for k, v in idmap_hier2_global.items()}

    idmap_hier2_fg = metadata['base_dataset_id_to_contiguous_id']
    idmap_hier2_fg_reversed = {v: k for k, v in idmap_hier2_fg.items()}
    hier2_fg_class_ids_global = [idmap_hier2_global[k] for k in idmap_hier2_fg.keys()]

    idmap_hier2_bg = metadata['novel_dataset_id_to_contiguous_id']
    idmap_hier2_bg_reversed = {v: k for k, v in idmap_hier2_bg.items()}
    hier2_bg_class_ids_global = [idmap_hier2_global[k] for k in idmap_hier2_bg.keys()]

    idmap_hier2_animal = {child_cat: i for i, child_cat in enumerate(hda_meta_info.super_cats_to_child_cats_idmap[hda_meta_info.super_cats_name2id['animal']])}
    idmap_hier2_animal_reversed = {v: k for k, v in idmap_hier2_animal.items()}
    hier2_animal_class_ids_global = [idmap_hier2_global[k] for k in idmap_hier2_animal.keys()]

    idmap_hier2_food = {child_cat: i for i, child_cat in enumerate(hda_meta_info.super_cats_to_child_cats_idmap[hda_meta_info.super_cats_name2id['food']])}
    idmap_hier2_food_reversed = {v: k for k, v in idmap_hier2_food.items()}
    hier2_food_class_ids_global = [idmap_hier2_global[k] for k in idmap_hier2_food.keys()]

    return {
        # TODO supplement the global id to name map (including bg) in case of plot 
        'idmap_hier1_global': idmap_hier1_global,
        'idmap_hier1_global_reversed': idmap_hier1_global_reversed,

        'idmap_hier2_global': idmap_hier2_global,
        'idmap_hier2_global_reversed': idmap_hier2_global_reversed,

        'idmap_hier2_fg': idmap_hier2_fg,
        'idmap_hier2_fg_reversed': idmap_hier2_fg_reversed,
        'hier2_fg_class_ids_global': hier2_fg_class_ids_global,

        'idmap_hier2_bg': idmap_hier2_bg,
        'idmap_hier2_bg_reversed': idmap_hier2_bg_reversed,
        'hier2_bg_class_ids_global': hier2_bg_class_ids_global,

        'idmap_hier2_animal': idmap_hier2_animal,
        'idmap_hier2_animal_reversed': idmap_hier2_animal_reversed,
        'hier2_animal_class_ids_global': hier2_animal_class_ids_global,

        'idmap_hier2_food': idmap_hier2_food,
        'idmap_hier2_food_reversed': idmap_hier2_food_reversed,
        'hier2_food_class_ids_global': hier2_food_class_ids_global,
    }

hda_meta_info = HDAMetaInfo()
metadata = hda_meta_info.get_meta_hda_all()
idmaps = init_idmaps(metadata, hda_meta_info)

labels_dict = dict({})
labels_dict['animal'] = idmaps['hier2_animal_class_ids_global']
labels_dict['food'] = idmaps['hier2_food_class_ids_global']
labels_dict['bg'] = idmaps['hier2_bg_class_ids_global']

# NOTE the following codes were used for distribution analysis
# pred_vs_true = torch.load("checkpoints_temp/coco_hda_cg_aug/rebuttal_10shot/pred_vs_true_10shot.pth")
# pred = pred_vs_true['pred']
# true = pred_vs_true['true']
# # the global contiguous ids of animal, food and bg child classes
# hier2_ids = labels_dict['animal']+labels_dict['food']+labels_dict['bg']
# mat = torch.zeros(len(hier2_ids), 43)
# for i, true_id in enumerate(hier2_ids):
#     mat[i] = torch.bincount(pred[true==true_id], minlength=43)

# mat_norm = torch.tensor(np.array(mat).astype(np.float)/np.array(mat).sum(axis=1)[:, np.newaxis])
# mat_row_cats = [hda_meta_info.child_cats_id2name[idmaps['idmap_hier2_global_reversed'][id]] for id in hier2_ids]
# mat_col_cats = [hda_meta_info.base_model_cats_id2name[idmaps['idmap_hier1_global_reversed'][id]] for id in range(42)]+['bg']

# ret = {"mat": mat, "mat_norm": mat_norm, "row_cats": mat_row_cats, "col_cats": mat_col_cats}
# torch.save(ret, "rebuttal_pred_class_refined_setting_10shot.pth")

# NOTE plot the predictions
key = 'Refined'
paths = {
    "Standard": "rebuttal_pred_class_standard_setting_10shot.pth",
    "Refined": "rebuttal_pred_class_refined_setting_10shot.pth",
    }

ret = torch.load(paths[key])

plot_params = {
                'figsize': (32,4),
                # 'figsize': (36,12),
                'annot': True,
                'linewidth': 0.3,
                'title': 'Base Classification {}'.format(key),
            }

mat = ret["mat"]
mat2 = torch.cat((mat[:,:40].sum(dim=1).reshape(mat.shape[0],-1),mat[:,40:]),dim=1)
mat2_norm = torch.tensor(np.array(mat2).astype(np.float)/np.array(mat2).sum(axis=1)[:, np.newaxis]).t()

plot = pd.DataFrame(np.around(np.array(mat2_norm), decimals=2), index=["ordinary classes", "animal", "food", "other/background"], columns=ret["row_cats"])
# plot = pd.DataFrame(np.around(np.array(ret["mat_norm"]), decimals=2), index=ret["row_cats"], columns=ret["col_cats"])
plt.figure(figsize = plot_params['figsize'])
# sn.set(font_scale=1.05)
hm = sn.heatmap(plot, annot=plot_params['annot'], linewidths=plot_params['linewidth'], cmap="YlGnBu", vmin=0, vmax=1.0, annot_kws={"size": 26 / np.sqrt(len(plot))}, cbar_kws={"pad":0.005})
hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize = 14, rotation=45)
hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize = 14, rotation=0)
plt.xlabel('Ground truth', fontsize = 15)
plt.ylabel('Base classification', fontsize = 15)
# plt.title(plot_params['title'], fontsize = 20)
plt.tight_layout()
plt.savefig('rebuttal_{}_v2.png'.format(plot_params['title']), dpi=300)

# NOTE the following codes were used to generate the confusion matrix plot
# labels_dict['hier1_all'] = list(idmaps['idmap_hier1_global'].values())
# labels_dict['hier2_all'] = list(idmaps['idmap_hier2_global'].values())

# label_names_dict = dict({})
# for k in ['animal', 'food', 'bg', 'hier1_all', 'hier2_all']:
#     if k == 'hier1_all':
#         label_names_dict[k] = [hda_meta_info.base_model_cats_id2name[idmaps["idmap_hier1_global_reversed"][label]] for label in labels_dict[k]]
#     else:
#         label_names_dict[k] = [hda_meta_info.child_cats_id2name[idmaps["idmap_hier2_global_reversed"][label]] for label in labels_dict[k]]

# preds_paths = dict({})
# preds_paths['HDA_base'] = "checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_hda_base/inference/instances_predictions.pth"
# preds_paths['HDA_1shot'] = "checkpoints_HDA_CG_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot_newfiltering_bbox/inference/instances_predictions.pth"
# preds_paths['HDA_5shot'] = "checkpoints_HDA_CG_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot_newfiltering_bbox/inference/instances_predictions.pth"
# preds_paths['HDA_10shot'] = "checkpoints_HDA_CG_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot_newfiltering_bbox/inference/instances_predictions.pth"
# preds_paths['TFA_1shot'] = "checkpoints_TFA_SGD_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot/inference/instances_predictions.pth"
# preds_paths['TFA_5shot'] = "checkpoints_TFA_SGD_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot/inference/instances_predictions.pth"
# preds_paths['TFA_10shot'] = "checkpoints_TFA_SGD_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot/inference/instances_predictions.pth"

# for model in ['HDA', 'TFA']:
#     for shot in [1,5,10]:
#         # for super_cat in ['animal', 'food', 'bg']:
#         super_cat = 'bg'
#         keys = ['{}_{}shot'.format(model, shot), super_cat, super_cat, super_cat]
#         # keys = ['{}_{}shot'.format(model, shot), 'hier2_all', 'hier2_all', 'hier2_all']
#         preds_path = preds_paths[keys[0]]
#         labels = labels_dict[keys[1]]
#         label_names_idx = label_names_dict[keys[2]]
#         # label_names_idx = label_names_dict['animal']+label_names_dict['food']
#         # label_names_idx = label_names_dict['bg']
#         label_names_col = label_names_dict[keys[3]]
#         plot_params = {
#             'figsize': (14,12),
#             'annot': True,
#             'linewidth': 0.3,
#             'title': '{}_{}'.format(keys[0], keys[1])
#         }

#         preds = torch.load(preds_path)
#         dts = []
#         gts = []
#         for pred in preds:
#             for obj in pred["instances"]:
#                 dts.append(obj["category_id"])
#                 gts.append(obj["category_id_gt"])
                
#         # NOTE modify the key dict here to be the old split, run the evaluation and save the matrices
#         mat = confusion_matrix(gts, dts, labels=labels)
#         plot = pd.DataFrame(mat, index = [i for i in label_names_idx],
#                         columns = [i for i in label_names_col])
#         plt.figure(figsize = plot_params['figsize'])
#         sn.heatmap(plot, annot=plot_params['annot'], linewidths=plot_params['linewidth'], cmap="YlGnBu",  vmin=0, vmax=14)
#         plt.xlabel('Pred', fontsize = 13)
#         plt.ylabel('True', fontsize = 13)
#         plt.title(plot_params['title'], fontsize = 14)
#         plt.savefig('checkpoints_temp/All_figs/Sep01_figs_CM/{}.png'.format(plot_params['title']))

