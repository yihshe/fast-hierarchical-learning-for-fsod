from fsdet.data.meta_coco_hda import HDAMetaInfo
from fsdet.data.lvis_v0_5_categories import LVIS_CATEGORIES
import torch

import argparse
import os

from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--src1', type=str, default='',
                        help='Path to the main checkpoint')
    parser.add_argument('--src2', type=str, default='',
                        help='Path to the secondary checkpoint (for combining)')
    parser.add_argument('--save-dir', type=str, default='',
                        help='Save directory')
    # Surgery method
    parser.add_argument('--method', choices=['combine', 'remove', 'randinit'],
                        required=True,
                        help='Surgery method. combine = '
                             'combine checkpoints. remove = for fine-tuning on '
                             'novel dataset, remove the final layer of the '
                             'base detector. randinit = randomly initialize '
                             'novel weights.')
    # Targets
    parser.add_argument('--param-name', type=str, nargs='+',
                        default=['roi_heads.box_predictor.cls_score',
                                 'roi_heads.box_predictor.bbox_pred'],
                        help='Target parameter names')
    parser.add_argument('--tar-name', type=str, default='model_reset',
                        help='Name of the new ckpt')
    # Dataset
    parser.add_argument('--coco', action='store_true',
                        help='For COCO models')
    parser.add_argument('--lvis', action='store_true',
                        help='For LVIS models')
    parser.add_argument('--coco-new-setting', action='store_true',
                        help='For COCO models TFA in the new setting (new split)')
    # Model architecture
    parser.add_argument('--two-stage-roi-heads', action='store_true',
                        help='For TwoStageROIHeads (i.e. HDAROIHeads in the standard setting)')
    parser.add_argument('--two-stage-roi-heads-new-setting', action='store_true',
                        help='For HDAROIHeads (in the new setting)')
    args = parser.parse_args()
    return args


def ckpt_surgery(args):
    """
    Either remove the final layer weights for fine-tuning on novel dataset or
    append randomly initialized weights for the novel classes.

    Note: The base detector for LVIS contains weights for all classes, but only
    the weights corresponding to base classes are updated during base training
    (this design choice has no particular reason). Thus, the random
    initialization step is not really necessary.
    """
    def surgery(param_name, is_weight, tar_size, ckpt, ckpt2=None):
        if not is_weight and param_name + '.bias' not in ckpt['model']:
            return
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        pretrained_weight = ckpt['model'][weight_name]
        prev_cls = pretrained_weight.size(0)
        # initialize the new weights
        if 'cls_score' in param_name:
            prev_cls -= 1
        if is_weight:
            feat_size = pretrained_weight.size(1)
            new_weight = torch.rand((tar_size, feat_size))
            torch.nn.init.normal_(new_weight, 0, 0.01)
        else:
            new_weight = torch.zeros(tar_size)
        # overwrite the new weights with corresponding pretrained weights
        if args.coco or args.lvis:
            for i, c in enumerate(BASE_CLASSES):
                idx = i if args.coco else c
                if 'cls_score' in param_name:
                    new_weight[IDMAP[c]] = pretrained_weight[idx]
                else:
                    new_weight[IDMAP[c]*4:(IDMAP[c]+1)*4] = \
                        pretrained_weight[idx*4:(idx+1)*4]

        elif args.coco_new_setting:
            for idx, c in enumerate(BASE_CLASSES):
                print(hda_meta_info.base_model_cats_id2name[c])
                if hda_meta_info.base_model_cats_id2name[c] not in ['animal', 'food']:
                    if 'cls_score' in param_name:
                        new_weight[IDMAP[c]] = pretrained_weight[idx]
                    else:
                        new_weight[IDMAP[c]*4:(IDMAP[c]+1)*4] = pretrained_weight[idx*4:(idx+1)*4]
                else:
                    if 'cls_score' in param_name:
                        pass
                    else:
                        for child_cat_id in hda_meta_info.super_cats_to_child_cats_idmap[c]:
                            new_weight[IDMAP[child_cat_id]*4:(IDMAP[child_cat_id]+1)*4] = pretrained_weight[idx*4:(idx+1)*4]

        else:
            new_weight[:prev_cls] = pretrained_weight[:prev_cls]

        if 'cls_score' in param_name:
            new_weight[-1] = pretrained_weight[-1]  # bg class
            
        ckpt['model'][weight_name] = new_weight

    surgery_loop(args, surgery)


def combine_ckpts(args):
    """
    Combine base detector with novel detector. Feature extractor weights are
    from the base detector. Only the final layer weights are combined.
    """
    def surgery(param_name, is_weight, tar_size, ckpt, ckpt2=None):
        if not is_weight and param_name + '.bias' not in ckpt['model']:
            return
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        pretrained_weight = ckpt['model'][weight_name]
        prev_cls = pretrained_weight.size(0)
        if 'cls_score' in param_name:
            prev_cls -= 1
        if is_weight:
            feat_size = pretrained_weight.size(1)
            new_weight = torch.rand((tar_size, feat_size))
        else:
            new_weight = torch.zeros(tar_size)
        if args.coco or args.lvis:
            for i, c in enumerate(BASE_CLASSES):
                idx = i if args.coco else c
                if 'cls_score' in param_name:
                    new_weight[IDMAP[c]] = pretrained_weight[idx]
                else:
                    new_weight[IDMAP[c]*4:(IDMAP[c]+1)*4] = \
                        pretrained_weight[idx*4:(idx+1)*4]
        else:
            new_weight[:prev_cls] = pretrained_weight[:prev_cls]

        ckpt2_weight = ckpt2['model'][weight_name]
        if args.coco or args.lvis:
            for i, c in enumerate(NOVEL_CLASSES):
                if 'cls_score' in param_name:
                    new_weight[IDMAP[c]] = ckpt2_weight[i]
                else:
                    new_weight[IDMAP[c]*4:(IDMAP[c]+1)*4] = \
                        ckpt2_weight[i*4:(i+1)*4]
            if 'cls_score' in param_name:
                new_weight[-1] = pretrained_weight[-1]
        else:
            if 'cls_score' in param_name:
                new_weight[prev_cls:-1] = ckpt2_weight[:-1]
                new_weight[-1] = pretrained_weight[-1]
            else:
                new_weight[prev_cls:] = ckpt2_weight
        ckpt['model'][weight_name] = new_weight
    
    surgery_loop(args, surgery)

def ckpt_two_stage(args):
    """
    Initialize the weights for RCNN with TwoStageROIHeads using either 'randinit' or 'combine' method.
    """
    def surgery(param_name, param_name_base, param_name_novel, is_weight, tar_size_base, tar_size_novel, ckpt, ckpt2=None):
        if not is_weight and param_name + '.bias' not in ckpt['model']:
            return
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        weight_name_base = param_name_base + ('.weight' if is_weight else '.bias')
        weight_name_novel = param_name_novel + ('.weight' if is_weight else '.bias')
        pretrained_weight = ckpt['model'][weight_name]
        
        # initialize the weights for base detector
        if args.lvis:
            if is_weight:
                feat_size = pretrained_weight.size(1)
                new_weight_base = torch.zeros((tar_size_base, feat_size))
            else:
                new_weight_base = torch.zeros(tar_size_base)
            
            for i, c in enumerate(BASE_CLASSES):
                if 'cls_score' in param_name:
                    new_weight_base[i] = pretrained_weight[c]
                else:
                    new_weight_base[i*4: (i+1)*4] = pretrained_weight[c*4: (c+1)*4]
            if 'cls_score' in param_name:
                new_weight_base[-1] = pretrained_weight[-1]
        else:
            new_weight_base = pretrained_weight
        
        # initalize the weights for novel detector
        if args.method == 'randinit':
            if is_weight:
                feat_size = pretrained_weight.size(1)
                new_weight_novel = torch.rand((tar_size_novel, feat_size))
                torch.nn.init.normal_(new_weight_novel, 0, 0.01)
            else:
                new_weight_novel = torch.zeros(tar_size_novel)
                
        elif args.method == 'combine':
            # NOTE experimental setting for novel classes containing both common and rare
            if args.lvis:
                ckpt2_weight = ckpt2['model'][weight_name]
                if is_weight:
                    feat_size = pretrained_weight.size(1)
                    new_weight_novel = torch.zeros((tar_size_novel, feat_size))
                else:
                    new_weight_novel = torch.zeros(tar_size_novel)
                
                # assign the pretrained common and rare weights to the novel detector
                for i,c in enumerate(NOVEL_CLASSES_COMMON):
                    if 'cls_score' in param_name:
                        new_weight_novel[IDMAP_NOVEL[c]] = pretrained_weight[c]
                    else:
                        new_weight_novel[IDMAP_NOVEL[c]*4: (IDMAP_NOVEL[c]+1)*4] = pretrained_weight[c*4: (c+1)*4]
                
                for i,c in enumerate(NOVEL_CLASSES_RARE):
                    if 'cls_score' in param_name:
                        new_weight_novel[IDMAP_NOVEL[c]] = ckpt2_weight[i]
                    else:
                        new_weight_novel[IDMAP_NOVEL[c]*4: (IDMAP_NOVEL[c]+1)*4] = ckpt2_weight[i*4: (i+1)*4]
                
                if 'cls_score' in param_name:
                    new_weight_novel[-1] = ckpt2_weight[-1]
            
            else:
                ckpt2_weight = ckpt2['model'][weight_name]
                new_weight_novel = ckpt2_weight

        ckpt['model'][weight_name_base] = new_weight_base
        ckpt['model'][weight_name_novel] = new_weight_novel
        
    surgery_loop_two_stage(args, surgery)

def surgery_loop_two_stage(args, surgery):
    assert args.method in ['combine', 'randinit'], "Initializetion for TwoStageROIHeads does not support 'remove'!"
    # Load checkpoints
    ckpt = torch.load(args.src1)
    if args.method == 'combine':
        ckpt2 = torch.load(args.src2)
        # NOTE previously the saved ckpt was named _ts.pth
        save_name = args.tar_name + '_combine_hda_std_setting_rc.pth'
    else:
        ckpt2 = None
        save_name = args.tar_name + '_' + 'surgery_hda_std_setting.pth'
    if args.save_dir == '':
        # By default, save to directory of src1
        save_dir = os.path.dirname(args.src1)
    else:
        save_dir = args.save_dir
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    reset_ckpt(ckpt)

    # Surgery
    if args.coco or args.lvis:
        tar_sizes_base = [len(BASE_CLASSES) + 1, len(BASE_CLASSES) * 4]
        tar_sizes_novel = [len(NOVEL_CLASSES) + 1, len(NOVEL_CLASSES) * 4]
    else:
        # voc
        len_base_classes_voc = 15
        len_novel_classes_voc = 5
        tar_sizes_base = [len_base_classes_voc + 1, len_base_classes_voc * 4]
        tar_sizes_novel = [len_novel_classes_voc + 1, len_novel_classes_voc * 4]
    param_names_base = ['roi_heads.box_predictor_base.cls_score', 'roi_heads.box_predictor_base.bbox_pred']
    param_names_novel = ['roi_heads.box_predictor.cls_score', 'roi_heads.box_predictor.bbox_pred']
    
    # Assume that base and novel predictors have the same model architecture
    for idx, (param_name, param_name_base, param_name_novel, tar_size_base, tar_size_novel) in enumerate(zip(args.param_name, param_names_base, param_names_novel, 
                                                                                                             tar_sizes_base, tar_sizes_novel)):
        surgery(param_name, param_name_base, param_name_novel, True, tar_size_base, tar_size_novel, ckpt, ckpt2)
        surgery(param_name, param_name_base, param_name_novel, False, tar_size_base, tar_size_novel, ckpt, ckpt2)

    # Save to file
    save_ckpt(ckpt, save_path)

def ckpt_two_stage_new_setting(args):
    """
    Initialize the weights for RCNN with TwoStageROIHeads using either 'randinit' or 'combine' method.
    """
    def surgery(
        param_name, param_name_base, param_name_bg, param_name_animal, param_name_food, is_weight, 
        tar_size_base, tar_size_bg, tar_size_animal, tar_size_food, ckpt
        ):
        if not is_weight and param_name + '.bias' not in ckpt['model']:
            return

        weight_name = param_name + ('.weight' if is_weight else '.bias')
        weight_name_base = param_name_base + ('.weight' if is_weight else '.bias')
        weight_name_bg = param_name_bg + ('.weight' if is_weight else '.bias')
        weight_name_animal = param_name_animal + ('.weight' if is_weight else '.bias')
        weight_name_food = param_name_food + ('.weight' if is_weight else '.bias')
        
        pretrained_weight = ckpt['model'][weight_name]
        new_weight_base = pretrained_weight

        assert args.method == 'randinit'
        
        if is_weight:
            feat_size = pretrained_weight.size(1)

            new_weight_bg = torch.rand((tar_size_bg, feat_size))
            torch.nn.init.normal_(new_weight_bg, 0, 0.01)
            if 'cls_score' in param_name:
                new_weight_animal = torch.rand((tar_size_animal, feat_size))
                torch.nn.init.normal_(new_weight_animal, 0, 0.01)
                new_weight_food = torch.rand((tar_size_food, feat_size))
                torch.nn.init.normal_(new_weight_food, 0, 0.01)
            else:
                new_weight_animal = pretrained_weight[animal_class_id*4:(animal_class_id+1)*4].repeat(tar_size_animal//4,1)
                new_weight_food = pretrained_weight[food_class_id*4:(food_class_id+1)*4].repeat(tar_size_food//4,1)
                
        else:
            new_weight_bg = torch.zeros(tar_size_bg)
            if 'cls_score' in param_name:
                new_weight_animal = torch.zeros(tar_size_animal)
                new_weight_food = torch.zeros(tar_size_food)
            else:
                new_weight_animal = pretrained_weight[animal_class_id*4:(animal_class_id+1)*4].repeat(tar_size_animal//4)
                new_weight_food = pretrained_weight[food_class_id*4:(food_class_id+1)*4].repeat(tar_size_food//4)

        # if 'cls_score' in param_name:
        #     new_weight_novel[-1] = new_weight_base[-1]
            
        ckpt['model'][weight_name_base] = new_weight_base
        ckpt['model'][weight_name_bg] = new_weight_bg
        ckpt['model'][weight_name_animal] = new_weight_animal
        ckpt['model'][weight_name_food] = new_weight_food
        
    surgery_loop_two_stage_new_setting(args, surgery)

def surgery_loop_two_stage_new_setting(args, surgery):
    assert args.method == 'randinit', "Initializetion for HDAROIHeads only supports 'randinit'!"
    # Load checkpoints
    ckpt = torch.load(args.src1)

    save_name = args.tar_name + '_' + 'surgery_ts_new_setting_bbox.pth'

    if args.save_dir == '':
        save_dir = os.path.dirname(args.src1)
    else:
        save_dir = args.save_dir
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    reset_ckpt(ckpt)

    num_classes = {
        'hier1_fg': 42,
        'hier2_fg': 60,
        'hier2_bg': 20,
        'hier2_animal': 10,
        'hier2_food': 10
    }

    # Surgery
    if args.coco:
        tar_sizes_base = [num_classes['hier1_fg'] + 1, num_classes['hier1_fg'] * 4]
        tar_sizes_bg = [num_classes['hier2_bg'] + 1, num_classes['hier2_bg'] * 4]
        tar_sizes_animal = [num_classes['hier2_animal'], num_classes['hier2_animal'] * 4]
        tar_sizes_food = [num_classes['hier2_food'], num_classes['hier2_food'] * 4]
    else:
        raise ValueError("HDAROIHead currently only supports COCO!")

    param_names_base = ['roi_heads.box_predictor_base.cls_score', 'roi_heads.box_predictor_base.bbox_pred']
    param_names_bg = ['roi_heads.box_predictor_bg.cls_score', 'roi_heads.box_predictor_bg.bbox_pred']
    param_names_animal = ['roi_heads.box_predictor_animal.cls_score', 'roi_heads.box_predictor_animal.bbox_pred']
    param_names_food = ['roi_heads.box_predictor_food.cls_score', 'roi_heads.box_predictor_food.bbox_pred']
    
    # Assume that base and novel predictors have the same model architecture
    for idx, (param_name, param_name_base, param_name_bg, param_name_animal, param_name_food, 
        tar_size_base, tar_size_bg, tar_size_animal, tar_size_food) in enumerate(zip(
        args.param_name, param_names_base, param_names_bg, param_names_animal, param_names_food,
        tar_sizes_base, tar_sizes_bg, tar_sizes_animal, tar_sizes_food)):
        surgery(
            param_name, param_name_base, param_name_bg, param_name_animal, param_name_food, True, 
            tar_size_base, tar_size_bg, tar_size_animal, tar_size_food, ckpt)
        surgery(
            param_name, param_name_base, param_name_bg, param_name_animal, param_name_food, False, 
            tar_size_base, tar_size_bg, tar_size_animal, tar_size_food, ckpt)

    # Save to file
    save_ckpt(ckpt, save_path)

def surgery_loop(args, surgery):
    # Load checkpoints
    ckpt = torch.load(args.src1)
    if args.method == 'combine':
        ckpt2 = torch.load(args.src2)
        save_name = args.tar_name + '_combine.pth'
    else:
        ckpt2 = None
        save_name = args.tar_name + '_' + \
            ('remove' if args.method == 'remove' else 'surgery') + '.pth'
            # ('remove' if args.method == 'remove' else 'surgery_new_setting') + '.pth'
    if args.save_dir == '':
        # By default, save to directory of src1
        save_dir = os.path.dirname(args.src1)
    else:
        save_dir = args.save_dir
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    reset_ckpt(ckpt)

    # Remove parameters
    if args.method == 'remove':
        for param_name in args.param_name:
            del ckpt['model'][param_name + '.weight']
            if param_name+'.bias' in ckpt['model']:
                del ckpt['model'][param_name+'.bias']
        save_ckpt(ckpt, save_path)
        return

    # Surgery
    tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]
    for idx, (param_name, tar_size) in enumerate(zip(args.param_name,
                                                     tar_sizes)):
        surgery(param_name, True, tar_size, ckpt, ckpt2)
        surgery(param_name, False, tar_size, ckpt, ckpt2)

    # Save to file
    save_ckpt(ckpt, save_path)


def save_ckpt(ckpt, save_name):
    torch.save(ckpt, save_name)
    print('save changed ckpt to {}'.format(save_name))


def reset_ckpt(ckpt):
    if 'scheduler' in ckpt:
        del ckpt['scheduler']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'iteration' in ckpt:
        ckpt['iteration'] = 0



if __name__ == '__main__':
    args = parse_args()

    # COCO
    # NOTE HDA to use the hda split for TFA models and initialization, the split needs to be added
    if args.coco:
        # COCO
        NOVEL_CLASSES = [
            1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67,
            72,
        ]
        BASE_CLASSES = [
            8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
            81, 82, 84, 85, 86, 87, 88, 89, 90,
        ]
        ALL_CLASSES = sorted(BASE_CLASSES + NOVEL_CLASSES)
        IDMAP = {v:i for i, v in enumerate(ALL_CLASSES)}
        TAR_SIZE = 80

    elif args.coco_new_setting:
        hda_meta_info = HDAMetaInfo()
        BASE_CLASSES = list(hda_meta_info.base_model_cats_id2name.keys())
        ALL_CLASSES = list(hda_meta_info.child_cats_id2name.keys())
        IDMAP = {v:i for i, v in enumerate(ALL_CLASSES)}
        TAR_SIZE = 80

    elif args.lvis:
        # LVIS
        # NOVEL_CLASSES = [
        #     0, 6, 9, 13, 14, 15, 20, 21, 30, 37, 38, 39, 41, 45, 48, 50, 51, 63,
        #     64, 69, 71, 73, 82, 85, 93, 99, 100, 104, 105, 106, 112, 115, 116,
        #     119, 121, 124, 126, 129, 130, 135, 139, 141, 142, 143, 146, 149,
        #     154, 158, 160, 162, 163, 166, 168, 172, 180, 181, 183, 195, 198,
        #     202, 204, 205, 208, 212, 213, 216, 217, 218, 225, 226, 230, 235,
        #     237, 238, 240, 241, 242, 244, 245, 248, 249, 250, 251, 252, 254,
        #     257, 258, 264, 265, 269, 270, 272, 279, 283, 286, 290, 292, 294,
        #     295, 297, 299, 302, 303, 305, 306, 309, 310, 312, 315, 316, 317,
        #     319, 320, 321, 323, 325, 327, 328, 329, 334, 335, 341, 343, 349,
        #     350, 353, 355, 356, 357, 358, 359, 360, 365, 367, 368, 369, 371,
        #     377, 378, 384, 385, 387, 388, 392, 393, 401, 402, 403, 405, 407,
        #     410, 412, 413, 416, 419, 420, 422, 426, 429, 432, 433, 434, 437,
        #     438, 440, 441, 445, 453, 454, 455, 461, 463, 468, 472, 475, 476,
        #     477, 482, 484, 485, 487, 488, 492, 494, 495, 497, 508, 509, 511,
        #     513, 514, 515, 517, 520, 523, 524, 525, 526, 529, 533, 540, 541,
        #     542, 544, 547, 550, 551, 552, 554, 555, 561, 563, 568, 571, 572,
        #     580, 581, 583, 584, 585, 586, 589, 591, 592, 593, 595, 596, 599,
        #     601, 604, 608, 609, 611, 612, 615, 616, 625, 626, 628, 629, 630,
        #     633, 635, 642, 644, 645, 649, 655, 657, 658, 662, 663, 664, 670,
        #     673, 675, 676, 682, 683, 685, 689, 695, 697, 699, 702, 711, 712,
        #     715, 721, 722, 723, 724, 726, 729, 731, 733, 734, 738, 740, 741,
        #     744, 748, 754, 758, 764, 766, 767, 768, 771, 772, 774, 776, 777,
        #     781, 782, 784, 789, 790, 794, 795, 796, 798, 799, 803, 805, 806,
        #     807, 808, 815, 817, 820, 821, 822, 824, 825, 827, 832, 833, 835,
        #     836, 840, 842, 844, 846, 856, 862, 863, 864, 865, 866, 868, 869,
        #     870, 871, 872, 875, 877, 882, 886, 892, 893, 897, 898, 900, 901,
        #     904, 905, 907, 915, 918, 919, 920, 921, 922, 926, 927, 930, 931,
        #     933, 939, 940, 944, 945, 946, 948, 950, 951, 953, 954, 955, 956,
        #     958, 959, 961, 962, 963, 969, 974, 975, 988, 990, 991, 998, 999,
        #     1001, 1003, 1005, 1008, 1009, 1010, 1012, 1015, 1020, 1022, 1025,
        #     1026, 1028, 1029, 1032, 1033, 1046, 1047, 1048, 1049, 1050, 1055,
        #     1066, 1067, 1068, 1072, 1073, 1076, 1077, 1086, 1094, 1099, 1103,
        #     1111, 1132, 1135, 1137, 1138, 1139, 1140, 1144, 1146, 1148, 1150,
        #     1152, 1153, 1156, 1158, 1165, 1166, 1167, 1168, 1169, 1171, 1178,
        #     1179, 1180, 1186, 1187, 1188, 1189, 1203, 1204, 1205, 1213, 1215,
        #     1218, 1224, 1225, 1227
        # ]
        # BASE_CLASSES = [c for c in range(1230) if c not in NOVEL_CLASSES]
        # ALL_CLASSES = sorted(BASE_CLASSES + NOVEL_CLASSES)
        # IDMAP = {v:i for i, v in enumerate(ALL_CLASSES)}
        # TAR_SIZE = 1230
        
        # NOTE experimental setting to contain both rare and common in novel classes
        lvis_categories_novel = [k for k in sorted(LVIS_CATEGORIES, key=lambda x: x["id"]) if k["frequency"] in ['c', 'r']]
        NOVEL_CLASSES = [k["id"]-1 for k in lvis_categories_novel]
        
        NOVEL_CLASSES_COMMON = [k["id"]-1 for k in lvis_categories_novel if k["frequency"]=="c"]
        NOVEL_CLASSES_RARE = [k["id"]-1 for k in lvis_categories_novel if k["frequency"]=="r"] 
        IDMAP_NOVEL = {v:i for i,v in enumerate(NOVEL_CLASSES)}

        BASE_CLASSES = [c for c in range(1230) if c not in NOVEL_CLASSES_RARE]
        ALL_CLASSES = sorted(BASE_CLASSES + NOVEL_CLASSES_RARE)
        IDMAP = {v:i for i, v in enumerate(ALL_CLASSES)}
        TAR_SIZE = 1230
    else:
        # VOC
        TAR_SIZE = 20


    if args.two_stage_roi_heads:
        ckpt_two_stage(args)
    elif args.two_stage_roi_heads_new_setting:
        hda_meta_info = HDAMetaInfo()
        animal_class_id = hda_meta_info.get_meta_hda_base()['base_dataset_id_to_contiguous_id'] [hda_meta_info.base_model_cats_name2id['animal']]
        food_class_id = hda_meta_info.get_meta_hda_base()['base_dataset_id_to_contiguous_id'] [hda_meta_info.base_model_cats_name2id['food']]
        ckpt_two_stage_new_setting(args)
    else:
        if args.method == 'combine':
            combine_ckpts(args)
        else:
            ckpt_surgery(args)
