import os
from subprocess import PIPE, STDOUT, Popen
import numpy as np
import random
import torch
from IPython import embed
import json
from fsdet.data.builtin_meta import _get_builtin_metadata

def run_cmd(cmd):
    p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    while True:
        line = p.stdout.readline().decode('utf-8')
        if not line:
            break
        print(line)

metadata = _get_builtin_metadata("coco_fewshot")
novel_ids = set(metadata["novel_dataset_id_to_contiguous_id"].keys())
base_ids = set(metadata["base_dataset_id_to_contiguous_id"].keys())
ids_to_names = {k: metadata["thing_classes"][metadata["thing_dataset_id_to_contiguous_id"][k]] for k in metadata["thing_dataset_id_to_contiguous_id"].keys()}

test_set = json.load(open("datasets/cocosplit/datasplit/5k.json"))
cats_imgs = {}
for anno in test_set["annotations"]:
    if anno["category_id"] not in set(cats_imgs.keys()):
        cats_imgs[anno["category_id"]]=list([])
        cats_imgs[anno["category_id"]].append(anno["image_id"])
    else:
        cats_imgs[anno["category_id"]].append(anno["image_id"])

cats_imgs_novel = {}
cats_imgs_base = {}
for k, v in cats_imgs.items():
    if k in novel_ids:
        cats_imgs_novel[ids_to_names[k]]=v
    else:
        cats_imgs_base[ids_to_names[k]]=v

# img_paths_novel = {}
# img_paths_base = {}
# for cat in random.sample(cats_imgs_novel.keys(),5):
# for cat in cats_imgs_novel.keys():
#     img_ids = random.sample(cats_imgs_novel[cat],10)
#     # img_paths_novel[cat] = []
#     img_paths_novel[cat] = ""
#     for img_id in img_ids:
#         path = "datasets/coco/val2014/COCO_val2014_{0:012}.jpg".format(img_id)
#         # img_paths_novel[cat].append(path)
#         img_paths_novel[cat]+=" {}".format(path)

# for cat in random.sample(cats_imgs_base.keys(),10):
#     img_ids = random.sample(cats_imgs_base[cat],5)
#     # img_paths_base[cat] = []
#     img_paths_base[cat]=""
#     for img_id in img_ids:
#         path = "datasets/coco/val2014/COCO_val2014_{0:012}.jpg".format(img_id)
#         # img_paths_base[cat].append(path)
#         img_paths_base[cat]+=" {}".format(path)


config_paths = {
    "TFA": "configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_10shot.yaml",
    "HDA": "configs/COCO-detection_HDA_CG_augmentation/faster_rcnn_R_101_FPN_ft_fc_all_10shot.yaml",
}
model_paths = {
    "TFA": "checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot/model_final.pth",
    "HDA": "checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot_hda_aug/model_final.pth",
    }

paths = ""
img_ids = [395463, 347390, 417846, 29308]
for img_id in img_ids:
    path = "datasets/coco/val2014/COCO_val2014_{0:012}.jpg".format(img_id)
    paths += " {}".format(path)

def main():
    # print("Novel\n")
    # for cat, paths in img_paths_novel.items():
    #     print(cat, "\n", paths, "\n")
    #     for model in ["TFA", "HDA"]:
    #         output_dir = "checkpoints_temp/figs/Nov15_visual0.25/{}/Novel/{}".format(model, cat)
    #         os.makedirs(output_dir, exist_ok=True)  
    #         demo_cmd = 'python3 -m demo.demo --confidence-threshold 0.25 ' \
    #                     '--config-file {} --input {} --output {} --opts MODEL.WEIGHTS {}'.format(config_paths[model], paths, output_dir, model_paths[model])
    #         run_cmd(demo_cmd)
            # embed()
    
    # print("Base\n")
    # for cat, paths in img_paths_base.items():
    #     print(cat, "\n", paths, "\n")
    #     for model in ["TFA", "HDA"]:
    #         output_dir = "checkpoints_temp/figs/Nov15_visual/{}/Base/{}".format(model, cat)
    #         os.makedirs(output_dir, exist_ok=True)  
    #         demo_cmd = 'python3 -m demo.demo ' \
    #                     '--config-file {} --input {} --output {} --opts MODEL.WEIGHTS {}'.format(config_paths[model], paths, output_dir, model_paths[model]) 
    #         run_cmd(demo_cmd)


    for model in ["TFA", "HDA"]:
        output_dir = "checkpoints_temp/figs/Nov16_visual/{}".format(model)
        os.makedirs(output_dir, exist_ok=True)  
        demo_cmd = 'python3 -m demo.demo --confidence-threshold 0.5 ' \
                    '--config-file {} --input {} --output {} --opts MODEL.WEIGHTS {}'.format(config_paths[model], paths, output_dir, model_paths[model])
        run_cmd(demo_cmd)

if __name__ == '__main__':
    main()