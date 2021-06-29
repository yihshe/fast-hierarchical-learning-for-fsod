import torch
from fsdet.modeling import GeneralizedRCNN

class GradientMask():
    """Mask the gradients for the pretrained weights"""
    def __init__(self, data_source: str, metadata = None, pretrained_params:dict = None):

        self.data_source = data_source
        self.NOVEL_CLASSES: list
        self.BASE_CLASSES: list
        self.IDMAP: dict
        self.TAR_SIZE: int
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # TODO add if data source is coco meta, pass the metadata of the task dataset to create the mask
        if data_source == 'coco':
            # COCO
            self.NOVEL_CLASSES = [
            1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67,
            72,
            ]
            self.BASE_CLASSES = [
                8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
                81, 82, 84, 85, 86, 87, 88, 89, 90,
            ]
            self.ALL_CLASSES = sorted(self.BASE_CLASSES + self.NOVEL_CLASSES)
            self.IDMAP = {v:i for i, v in enumerate(self.ALL_CLASSES)}
            self.TAR_SIZE = 80    
         
        elif data_source == 'coco_meta_all_tfa':
            # novel classes, base classes, tar_sizem, and base tar size
            assert metadata is not None, "Metadata of the task is missing!"
            self.NOVEL_CLASSES = list(metadata.novel_dataset_id_to_contiguous_id.keys())
            self.BASE_CLASSES = list(metadata.base_dataset_id_to_contiguous_id.keys())
            self.ALL_CLASSES = list(metadata.thing_dataset_id_to_contiguous_id.keys())
            
            self.IDMAP = metadata.thing_dataset_id_to_contiguous_id
            
            self.TAR_SIZE = len(self.ALL_CLASSES)
            self.BASE_TAR_SIZE = len(self.BASE_CLASSES)

            self.pretrained_params = pretrained_params
        
        elif data_source =='coco_meta_novel':
            assert metadata is not None, "Metadata of the task is missing!"
            # NOTE the thing dataset id has been overwritten by the novel dataset id as the scope is novel when sampling task
            self.NOVEL_CLASSES = list(metadata.novel_dataset_id_to_contiguous_id.keys())
            self.IDMAP = metadata.novel_dataset_id_to_contiguous_id
            self.TAR_SIZE = len(self.NOVEL_CLASSES)
            self.pretrained_params = pretrained_params

        elif data_source == 'lvis':
            # LVIS
            self.NOVEL_CLASSES = [
                0, 6, 9, 13, 14, 15, 20, 21, 30, 37, 38, 39, 41, 45, 48, 50, 51, 63,
                64, 69, 71, 73, 82, 85, 93, 99, 100, 104, 105, 106, 112, 115, 116,
                119, 121, 124, 126, 129, 130, 135, 139, 141, 142, 143, 146, 149,
                154, 158, 160, 162, 163, 166, 168, 172, 180, 181, 183, 195, 198,
                202, 204, 205, 208, 212, 213, 216, 217, 218, 225, 226, 230, 235,
                237, 238, 240, 241, 242, 244, 245, 248, 249, 250, 251, 252, 254,
                257, 258, 264, 265, 269, 270, 272, 279, 283, 286, 290, 292, 294,
                295, 297, 299, 302, 303, 305, 306, 309, 310, 312, 315, 316, 317,
                319, 320, 321, 323, 325, 327, 328, 329, 334, 335, 341, 343, 349,
                350, 353, 355, 356, 357, 358, 359, 360, 365, 367, 368, 369, 371,
                377, 378, 384, 385, 387, 388, 392, 393, 401, 402, 403, 405, 407,
                410, 412, 413, 416, 419, 420, 422, 426, 429, 432, 433, 434, 437,
                438, 440, 441, 445, 453, 454, 455, 461, 463, 468, 472, 475, 476,
                477, 482, 484, 485, 487, 488, 492, 494, 495, 497, 508, 509, 511,
                513, 514, 515, 517, 520, 523, 524, 525, 526, 529, 533, 540, 541,
                542, 544, 547, 550, 551, 552, 554, 555, 561, 563, 568, 571, 572,
                580, 581, 583, 584, 585, 586, 589, 591, 592, 593, 595, 596, 599,
                601, 604, 608, 609, 611, 612, 615, 616, 625, 626, 628, 629, 630,
                633, 635, 642, 644, 645, 649, 655, 657, 658, 662, 663, 664, 670,
                673, 675, 676, 682, 683, 685, 689, 695, 697, 699, 702, 711, 712,
                715, 721, 722, 723, 724, 726, 729, 731, 733, 734, 738, 740, 741,
                744, 748, 754, 758, 764, 766, 767, 768, 771, 772, 774, 776, 777,
                781, 782, 784, 789, 790, 794, 795, 796, 798, 799, 803, 805, 806,
                807, 808, 815, 817, 820, 821, 822, 824, 825, 827, 832, 833, 835,
                836, 840, 842, 844, 846, 856, 862, 863, 864, 865, 866, 868, 869,
                870, 871, 872, 875, 877, 882, 886, 892, 893, 897, 898, 900, 901,
                904, 905, 907, 915, 918, 919, 920, 921, 922, 926, 927, 930, 931,
                933, 939, 940, 944, 945, 946, 948, 950, 951, 953, 954, 955, 956,
                958, 959, 961, 962, 963, 969, 974, 975, 988, 990, 991, 998, 999,
                1001, 1003, 1005, 1008, 1009, 1010, 1012, 1015, 1020, 1022, 1025,
                1026, 1028, 1029, 1032, 1033, 1046, 1047, 1048, 1049, 1050, 1055,
                1066, 1067, 1068, 1072, 1073, 1076, 1077, 1086, 1094, 1099, 1103,
                1111, 1132, 1135, 1137, 1138, 1139, 1140, 1144, 1146, 1148, 1150,
                1152, 1153, 1156, 1158, 1165, 1166, 1167, 1168, 1169, 1171, 1178,
                1179, 1180, 1186, 1187, 1188, 1189, 1203, 1204, 1205, 1213, 1215,
                1218, 1224, 1225, 1227
            ]
            self.BASE_CLASSES = [c for c in range(1230) if c not in self.NOVEL_CLASSES]
            self.ALL_CLASSES = sorted(self.BASE_CLASSES + self.NOVEL_CLASSES)
            self.IDMAP = {v:i for i, v in enumerate(self.ALL_CLASSES)}
            self.TAR_SIZE = 1230
        
        else:
            # VOC
            self.TAR_SIZE = 20


    def create_meta_setup(self, param_names = ['roi_heads.box_predictor.cls_score', 'roi_heads.box_predictor.bbox_pred']):
        assert self.data_source in ['coco_meta_all_tfa', 'coco_meta_novel'], "Please correct the task type for generating meta data!"
        
        if self.data_source is 'coco_meta_all_tfa':
            return self.create_meta_setup_tfa(param_names = param_names)
        elif self.data_source is 'coco_meta_novel':
            return self.create_meta_setup_novel(param_names = param_names)

    def create_meta_setup_tfa(self, param_names = ['roi_heads.box_predictor.cls_score', 'roi_heads.box_predictor.bbox_pred']):
        masks = list([])
        randinit_params = list([])
        base_params = dict({})

        results = list([])
        tar_sizes = [self.TAR_SIZE+1, self.TAR_SIZE*4]
        base_tar_sizes = [self.BASE_TAR_SIZE+1, self.BASE_TAR_SIZE*4]

        for idx, (param_name, tar_size, base_tar_size) in enumerate(zip(param_names, tar_sizes, base_tar_sizes)):
            for is_weight in [True, False]:
                 results.append(self.single_meta_setup_tfa(param_name, tar_size, base_tar_size, is_weight, self.pretrained_params))

        results = [result for result in results if result is not None]
        for result in results:
            masks.append(result['mask'].to(self.device))
            randinit_params.append(result['randinit_weight'].to(self.device))
            base_params[result['weight_name']] = result['base_weight'].to(self.device)

        return {
                'masks': masks, 
                'randinit_params': randinit_params, 
                'base_params': base_params, 
                'novel_classes': self.NOVEL_CLASSES,
                'id_map': self.IDMAP
                }
    
    def single_meta_setup_tfa(self, param_name: str, tar_size: int, base_tar_size: int, is_weight: bool, params: dict):
        if not is_weight and param_name+'.bias' not in params.keys():
            return
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        pretrained_weight = params[weight_name]
        if is_weight:
            feat_size = pretrained_weight.shape[1]
            mask = torch.ones((tar_size, feat_size))
            randinit_weight = torch.rand((tar_size, feat_size))
            torch.nn.init.normal_(randinit_weight, 0, 0.01)
            base_weight = torch.zeros((base_tar_size, feat_size))
        else:
            mask = torch.ones(tar_size)
            randinit_weight = torch.zeros(tar_size)
            base_weight = torch.zeros(base_tar_size)

        for idx, c in enumerate(self.BASE_CLASSES):
            if 'cls_score' in param_name:
                mask[self.IDMAP[c]] = 0
                randinit_weight[self.IDMAP[c]] = pretrained_weight[self.IDMAP[c]]
                base_weight[idx] = pretrained_weight[self.IDMAP[c]]
            else:
                mask[self.IDMAP[c]*4:(self.IDMAP[c]+1)*4] = 0
                randinit_weight[self.IDMAP[c]*4:(self.IDMAP[c]+1)*4] = pretrained_weight[self.IDMAP[c]*4:(self.IDMAP[c]+1)*4]
                base_weight[idx*4:(idx+1)*4] = pretrained_weight[self.IDMAP[c]*4:(self.IDMAP[c]+1)*4]
        
        # for background classes
        if 'cls_score' in param_name:
            mask[-1] = 0
            randinit_weight[-1] = pretrained_weight[-1]
            base_weight[-1] = pretrained_weight[-1]

        result = {'weight_name': weight_name,
                  'mask': mask,
                  'randinit_weight': randinit_weight,
                  'base_weight': base_weight}

        return result

    def create_meta_setup_novel(self, param_names = ['roi_heads.box_predictor.cls_score', 'roi_heads.box_predictor.bbox_pred']):
        randinit_params = list([])

        results = list([])
        tar_sizes = [self.TAR_SIZE+1, self.TAR_SIZE*4]

        for idx, (param_name, tar_size) in enumerate(zip(param_names, tar_sizes)):
            for is_weight in [True, False]:
                 results.append(self.single_meta_setup_novel(param_name, tar_size, is_weight, self.pretrained_params))

        results = [result for result in results if result is not None]
        for result in results:
            randinit_params.append(result['randinit_weight'].to(self.device))

        return {
                'randinit_params': randinit_params, 
                'novel_classes': self.NOVEL_CLASSES,
                'id_map': self.IDMAP
                }
    
    def single_meta_setup_novel(self, param_name: str, tar_size: int, is_weight: bool, params: dict):
        if not is_weight and param_name+'.bias' not in params.keys():
            return
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        pretrained_weight = params[weight_name]
        if is_weight:
            feat_size = pretrained_weight.shape[1]
            randinit_weight = torch.rand((tar_size, feat_size))
            torch.nn.init.normal_(randinit_weight, 0, 0.01)
        else:
            randinit_weight = torch.zeros(tar_size)

        result = {'weight_name': weight_name,
                  'randinit_weight': randinit_weight,
                 }

        return result

    def create_mask(self, params: dict, base_params: dict,
        param_names = ['roi_heads.box_predictor.cls_score', 'roi_heads.box_predictor.bbox_pred']):
        """ 
        Generate a gradient mask for the pretrained weights.
        """
        masks = list([])
        # params = model.state_dict()
        tar_sizes = [self.TAR_SIZE+1, self.TAR_SIZE*4]
        for idx, (param_name, tar_size) in enumerate(zip(param_names, tar_sizes)):
            masks.append(self.single_mask(param_name, tar_size, True, params, base_params))
            masks.append(self.single_mask(param_name, tar_size, False, params, base_params))
        
        masks = [i.to(self.device) for i in masks if i is not None]
        return masks
        
    def single_mask(self, param_name: str, tar_size: int, is_weight: bool, params: dict, base_params: dict):
        """
        Creare mask for a single layer.
        """
        if not is_weight and param_name+'.bias' not in params.keys():
            return
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        prev_cls = base_params[weight_name].size(0)

        if 'cls_score' in param_name:
            prev_cls -= 1

        # TODO here the new weight can also be initialized with the mask (in a separate part for meta)
        # the rand init weights will be loaded to the model (cosine)
        # and return base params (dict) at the same time (mask, new weights, base params)
        mask = torch.ones(params[weight_name].shape)

        if self.data_source == 'coco' or self.data_source == 'lvis':
            for i, c in enumerate(self.BASE_CLASSES):
                if 'cls_score' in param_name:
                    mask[self.IDMAP[c]] = 0
                else:
                    mask[self.IDMAP[c]*4:(self.IDMAP[c]+1)*4] = 0
        else:
            # for voc
            mask[:prev_cls] = 0
        
        if 'cls_score' in param_name:
            mask[-1] = 0
        
        return mask




