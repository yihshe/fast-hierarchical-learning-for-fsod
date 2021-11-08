import json
import os
from IPython import embed

# NOTE this dict only contain the isthing child class (80 in total)
class HDAMetaInfo:
    def __init__(self):

        cats = [{"supercategory": "person", "id": 1, "name": "person"}, {"supercategory": "vehicle", "id": 2, "name": "bicycle"}, {"supercategory": "vehicle", "id": 3, "name": "car"}, {"supercategory": "vehicle", "id": 4, "name": "motorcycle"}, {"supercategory": "vehicle", "id": 5, "name": "airplane"}, {"supercategory": "vehicle", "id": 6, "name": "bus"}, {"supercategory": "vehicle", "id": 7, "name": "train"}, {"supercategory": "vehicle", "id": 8, "name": "truck"}, {"supercategory": "vehicle", "id": 9, "name": "boat"}, {"supercategory": "outdoor", "id": 10, "name": "traffic light"}, {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"}, {"supercategory": "outdoor", "id": 13, "name": "stop sign"}, {"supercategory": "outdoor", "id": 14, "name": "parking meter"}, {"supercategory": "outdoor", "id": 15, "name": "bench"}, {"supercategory": "animal", "id": 16, "name": "bird"}, {"supercategory": "animal", "id": 17, "name": "cat"}, {"supercategory": "animal", "id": 18, "name": "dog"}, {"supercategory": "animal", "id": 19, "name": "horse"}, {"supercategory": "animal", "id": 20, "name": "sheep"}, {"supercategory": "animal", "id": 21, "name": "cow"}, {"supercategory": "animal", "id": 22, "name": "elephant"}, {"supercategory": "animal", "id": 23, "name": "bear"}, {"supercategory": "animal", "id": 24, "name": "zebra"}, {"supercategory": "animal", "id": 25, "name": "giraffe"}, {"supercategory": "accessory", "id": 27, "name": "backpack"}, {"supercategory": "accessory", "id": 28, "name": "umbrella"}, {"supercategory": "accessory", "id": 31, "name": "handbag"}, {"supercategory": "accessory", "id": 32, "name": "tie"}, {"supercategory": "accessory", "id": 33, "name": "suitcase"}, {"supercategory": "sports", "id": 34, "name": "frisbee"}, {"supercategory": "sports", "id": 35, "name": "skis"}, {"supercategory": "sports", "id": 36, "name": "snowboard"}, {"supercategory": "sports", "id": 37, "name": "sports ball"}, {"supercategory": "sports", "id": 38, "name": "kite"}, {"supercategory": "sports", "id": 39, "name": "baseball bat"}, {"supercategory": "sports", "id": 40, "name": "baseball glove"}, {"supercategory": "sports", "id": 41, "name": "skateboard"}, {"supercategory": "sports", "id": 42, "name": "surfboard"}, {"supercategory": "sports", "id": 43, "name": "tennis racket"}, {"supercategory": "kitchen", "id": 44, "name": "bottle"}, {"supercategory": "kitchen", "id": 46, "name": "wine glass"}, {"supercategory": "kitchen", "id": 47, "name": "cup"}, {"supercategory": "kitchen", "id": 48, "name": "fork"}, {"supercategory": "kitchen", "id": 49, "name": "knife"}, {"supercategory": "kitchen", "id": 50, "name": "spoon"}, {"supercategory": "kitchen", "id": 51, "name": "bowl"}, {"supercategory": "food", "id": 52, "name": "banana"}, {"supercategory": "food", "id": 53, "name": "apple"}, {"supercategory": "food", "id": 54, "name": "sandwich"}, {"supercategory": "food", "id": 55, "name": "orange"}, {"supercategory": "food", "id": 56, "name": "broccoli"}, {"supercategory": "food", "id": 57, "name": "carrot"}, {"supercategory": "food", "id": 58, "name": "hot dog"}, {"supercategory": "food", "id": 59, "name": "pizza"}, {"supercategory": "food", "id": 60, "name": "donut"}, {"supercategory": "food", "id": 61, "name": "cake"}, {"supercategory": "furniture", "id": 62, "name": "chair"}, {"supercategory": "furniture", "id": 63, "name": "couch"}, {"supercategory": "furniture", "id": 64, "name": "potted plant"}, {"supercategory": "furniture", "id": 65, "name": "bed"}, {"supercategory": "furniture", "id": 67, "name": "dining table"}, {"supercategory": "furniture", "id": 70, "name": "toilet"}, {"supercategory": "electronic", "id": 72, "name": "tv"}, {"supercategory": "electronic", "id": 73, "name": "laptop"}, {"supercategory": "electronic", "id": 74, "name": "mouse"}, {"supercategory": "electronic", "id": 75, "name": "remote"}, {"supercategory": "electronic", "id": 76, "name": "keyboard"}, {"supercategory": "electronic", "id": 77, "name": "cell phone"}, {"supercategory": "appliance", "id": 78, "name": "microwave"}, {"supercategory": "appliance", "id": 79, "name": "oven"}, {"supercategory": "appliance", "id": 80, "name": "toaster"}, {"supercategory": "appliance", "id": 81, "name": "sink"}, {"supercategory": "appliance", "id": 82, "name": "refrigerator"}, {"supercategory": "indoor", "id": 84, "name": "book"}, {"supercategory": "indoor", "id": 85, "name": "clock"}, {"supercategory": "indoor", "id": 86, "name": "vase"}, {"supercategory": "indoor", "id": 87, "name": "scissors"}, {"supercategory": "indoor", "id": 88, "name": "teddy bear"}, {"supercategory": "indoor", "id": 89, "name": "hair drier"}, {"supercategory": "indoor", "id": 90, "name": "toothbrush"}]
        super_cats_to_child_cats = dict({})
        for cat in cats:
            if cat["supercategory"] not in super_cats_to_child_cats.keys():
                super_cats_to_child_cats[cat["supercategory"]] = dict({cat['id']: cat['name']})
            else:
                super_cats_to_child_cats[cat["supercategory"]][cat['id']] = cat['name']

        super_cats = [k for k in super_cats_to_child_cats.keys()]
        novel_super_cats = ['person', 'vehicle', 'accessory', 'furniture']
        base_super_cats = [k for k in super_cats_to_child_cats.keys() if k not in novel_super_cats]
        super_cats_id2name = {1000+i: k for i, k in enumerate(super_cats_to_child_cats.keys())}
        super_cats_name2id = {v:k for k,v in super_cats_id2name.items()}

        child_cats = [k['name'] for k in cats]
        novel_child_cats = [k['name'] for k in cats if k['supercategory'] in novel_super_cats]
        base_child_cats = [k['name'] for k in cats if k['supercategory'] in base_super_cats]
        base_animal_child_cats = [k['name'] for k in cats if k['supercategory'] == 'animal']
        base_food_child_cats = [k['name'] for k in cats if k['supercategory'] == 'food']
        base_other_child_cats = [k['name'] for k in cats if k['supercategory'] in base_super_cats and k['supercategory'] not in ['animal', 'food']]
        child_cats_id2name = {k['id']:k['name'] for k in cats}
        child_cats_name2id = {v:k for k,v in child_cats_id2name.items()}
        
        super_cats_to_child_cats_idmap = dict({})
        for i, super_cat in enumerate(super_cats):
            super_cat_id = 1000+i
            super_cats_to_child_cats_idmap[super_cat_id] = list(super_cats_to_child_cats[super_cat].keys())
        
        child_cats_to_super_cats_idmap = dict({})
        for super_cat_id in super_cats_to_child_cats_idmap.keys():
            child_cat_ids = super_cats_to_child_cats_idmap[super_cat_id]
            for child_cat_id in child_cat_ids:
                child_cats_to_super_cats_idmap[child_cat_id] = super_cat_id
        
        base_model_cats_name2id = dict({})
        for child_cat in base_child_cats:
            child_cat_id = child_cats_name2id[child_cat]
            super_cat_id = child_cats_to_super_cats_idmap[child_cat_id]
            if super_cats_id2name[super_cat_id] in ['animal', 'food']:
                continue
            base_model_cats_name2id[child_cat] = child_cat_id
        for super_cat in ['animal', 'food']:
            base_model_cats_name2id[super_cat] = super_cats_name2id[super_cat]
        base_model_cats_id2name = {v:k for k,v in base_model_cats_name2id.items()}

        self.super_cats = super_cats
        self.novel_super_cats = novel_super_cats
        self.base_super_cats = base_super_cats
        self.super_cats_id2name = super_cats_id2name
        self.super_cats_name2id = super_cats_name2id

        self.child_cats = child_cats
        self.novel_child_cats = novel_child_cats
        self.base_child_cats = base_child_cats
        self.base_animal_child_cats = base_animal_child_cats
        self.base_food_child_cats = base_food_child_cats
        self.base_other_child_cats = base_other_child_cats
        self.child_cats_id2name = child_cats_id2name
        self.child_cats_name2id = child_cats_name2id

        self.super_cats_to_child_cats_idmap = super_cats_to_child_cats_idmap
        self.child_cats_to_super_cats_idmap = child_cats_to_super_cats_idmap

        self.base_model_cats_name2id = base_model_cats_name2id
        self.base_model_cats_id2name = base_model_cats_id2name
    
    def get_meta_hda_all(self):
        metadata = dict({})
        metadata["thing_classes"] = self.child_cats
        metadata["thing_dataset_id_to_contiguous_id"] = {self.child_cats_name2id[child_cat]: i for i, child_cat in enumerate(self.child_cats)}
        metadata["novel_classes"] = self.novel_child_cats
        metadata["novel_dataset_id_to_contiguous_id"] = {self.child_cats_name2id[child_cat]: i for i, child_cat in enumerate(self.novel_child_cats)}
        metadata["base_classes"] = self.base_child_cats
        metadata["base_dataset_id_to_contiguous_id"] = {self.child_cats_name2id[child_cat]: i for i, child_cat in enumerate(self.base_child_cats)}

        return metadata

    def get_meta_hda_base(self):
        metadata = dict({})
        metadata["base_classes"] = list(self.base_model_cats_name2id.keys())
        metadata["base_dataset_id_to_contiguous_id"] = {k: i for i, k in enumerate(self.base_model_cats_id2name.keys())}
        return metadata

    def dataset_id_mapping(self, source_dir = "datasets/cocosplit/datasplit", source_file = "5k.json"):
        # NOTE map the category id in the original test set for evaluating the base detector trained in hierarchical setting
        dataset = json.load(open(os.path.join(source_dir, source_file)))
        for anno in dataset['annotations']:
            super_cat_id = self.child_cats_to_super_cats_idmap[anno['category_id']]
            if self.super_cats_id2name[super_cat_id] in ['animal', 'food']:
                anno['category_id'] = super_cat_id
        with open(os.path.join(source_dir, '{}_hda_base.json'.format(source_file.split('.')[0])), "w") as fp:
            json.dump(dataset, fp)






