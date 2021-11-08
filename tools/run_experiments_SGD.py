import argparse
import os
from IPython.terminal.embed import embed
import yaml
from ast import literal_eval as make_tuple
from subprocess import PIPE, STDOUT, Popen


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 3, 5, 10],
                        help='Shots to run experiments over')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 20],
                        help='Range of seeds to run')
    parser.add_argument('--root', type=str, default='./', help='Root of data')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of path')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    # TODO modify the ckpt freq
    parser.add_argument('--ckpt-freq', type=int, default=10,
                        help='Frequency of saving checkpoints')
    # Model
    parser.add_argument('--HDA', action='store_true',
                        help='Hierarchial detection approach')
    parser.add_argument('--fc', action='store_true',
                        help='Model uses FC instead of cosine')
    parser.add_argument('--two-stage', action='store_true',
                        help='Two-stage fine-tuning')
    parser.add_argument('--novel-finetune', action='store_true',
                        help='Fine-tune novel weights first')
    parser.add_argument('--unfreeze', action='store_true',
                        help='Unfreeze feature extractor')
    # PASCAL arguments
    parser.add_argument('--split', '-s', type=int, default=1, help='Data split')
    # COCO arguments
    parser.add_argument('--coco', action='store_true', help='Use COCO dataset')
    parser.add_argument('--eval-saved-results', action = 'store_true', help = 'Evaluate saved results in the new setting')

    args = parser.parse_args()
    return args


def load_yaml_file(fname):
    with open(fname, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_cmd(cmd):
    p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    while True:
        line = p.stdout.readline().decode('utf-8')
        if not line:
            break
        print(line)


def run_exp(cfg, configs, eval_saved_results = False):
    """
    Run training and evaluation scripts based on given config files.
    """
    if eval_saved_results is False:
        # Train
        output_dir = configs['OUTPUT_DIR']
        model_path = os.path.join(args.root, output_dir, 'model_final.pth')
        if not os.path.exists(model_path):
            train_cmd = 'python -m tools.train_net_modified --dist-url auto --num-gpus {} ' \
                        '--config-file {} --resume'.format(args.num_gpus, cfg)
            run_cmd(train_cmd)
        if not args.novel_finetune:
            # Test
            res_path = os.path.join(args.root, output_dir, 'inference',
                                    'res_final.json')
            if not os.path.exists(res_path):
                test_cmd = 'python -m tools.test_net --dist-url auto --num-gpus {} ' \
                        '--config-file {} --resume --eval-only'.format(args.num_gpus,
                                                                        cfg)
                run_cmd(test_cmd)
    else:
        results_path = os.path.join(args.root, configs['OUTPUT_DIR'], "inference", "coco_instances_results.json")
        assert os.path.exists(results_path)
        test_cmd = 'python -m tools.test_net --dist-url auto --num-gpus {} ' \
                    '--config-file {} --results-path {} --resume --eval-only'.format(args.num_gpus,
                                                                    cfg, results_path)
        run_cmd(test_cmd)

def get_config(seed, shot):
    """
    For a given seed and shot, generate a config file based on a template
    config file that is used for training/evaluation.
    You can extend/modify this function to fit your use-case.
    """
    # NOTE fix and shot first and iterate over seed
    # The TFA with CG for COCO now only supports randominit weight with fixed number of iterations
    model = 'HDA' if args.HDA else 'TFA'
    if args.coco:
        # COCO
        # assert not args.two_stage, 'Only supports random weights for COCO now in TFA with CG'
        assert args.two_stage, 'Only supports novel weights for COCO now'

        if args.novel_finetune:
            # Fine-tune novel classifier
            ITERS = {
                # NOTE step and max iter
                1: (10000, 500),
                2: (10000, 1500),
                3: (10000, 1500),
                5: (10000, 1500),
                10: (10000, 2000),
                30: (10000, 6000),
            }
            mode = 'novel'

            assert not args.fc and not args.unfreeze
        else:
            # Fine-tune entire classifier
            ITERS = {
                1: (14400, 16000),
                2: (28800, 32000),
                3: (43200, 48000),
                5: (72000, 80000),
                10: (144000, 160000),
                30: (216000, 240000),
            }
            mode = 'all'

        split = temp_split = ''
        temp_mode = mode

        # NOTE to be modified the path of config and the path for output
        config_dir = 'configs/thesis/COCO-detection_{}_SGD_new_setting'.format(model)
        ckpt_dir = 'checkpoints/thesis/checkpoints_{}_SGD_new_setting/coco/faster_rcnn'.format(model)
        # TODO the path needs to be checked for seed 0
        base_cfg = '../../../Base-RCNN-FPN.yaml' if seed!=0 else '../../Base-RCNN-FPN.yaml'
    else:
        # PASCAL VOC
        assert not args.two_stage, 'Only supports random weights for PASCAL now'

        # TODO determine the number of Newton and CG iterations
        # for SGD the number of iterations needs to be changed
        ITERS = {
            1: (3500, 4000),
            2: (7000, 8000),
            3: (10500, 12000),
            5: (17500, 20000),
            10: (35000, 40000),
        }

        split = 'split{}'.format(args.split)
        mode = 'all{}'.format(args.split)
        temp_split = 'split1'
        temp_mode = 'all1'
        
        # NOTE where the template config file should be stored in config_dir (1shot)
        config_dir = 'configs/PascalVOC-detection_{}_SGD'.format(model)
        ckpt_dir = 'checkpoints_{}_SGD/voc/faster_rcnn'.format(model)
        base_cfg = '../../../Base-RCNN-FPN.yaml' if seed!=0 else '../../Base-RCNN-FPN.yaml'

    seed_str = 'seed{}'.format(seed) if seed != 0 else ''
    fc = '_fc' if args.fc else ''
    unfreeze = '_unfreeze' if args.unfreeze else ''
    # Read an example config file for the config parameters
    # NOTE the example config file needs to be edited for CG or HDA
    # NOTE temp_mode, novel, all, all1 as template
    temp = os.path.join(
        temp_split, 'faster_rcnn_R_101_FPN_ft{}_{}_1shot{}'.format(
            fc, temp_mode, unfreeze)
    )
    config = os.path.join(args.root, config_dir, temp + '.yaml')

    # NOTE as the folder name for output, the number of shot needs to be changed
    prefix = 'faster_rcnn_R_101_FPN_ft{}_{}_{}shot{}{}'.format(
        fc, mode, shot, unfreeze, args.suffix)

    # NOTE dir to output the model ckpt
    # every seed folder in ckpt will contain the result folder for different shot (named by prefix)
    # seed0 (seed1 seed2 ...), all1_shot1 2 3, all2_shot 1 2 3 
    output_dir = os.path.join(args.root, ckpt_dir, seed_str)
    os.makedirs(output_dir, exist_ok=True)
    # NOTE dir to save the config will contain the cfg for different shot 
    # every seed folder in configs will contain the 
    # split1, seed0 (seed1 seed2 ...), all1_shot1 all1_shot2 ...
    save_dir = os.path.join(
        args.root, config_dir, split, seed_str,
    )
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, prefix + '.yaml')

    configs = load_yaml_file(config)
    # TODO modify here also aggrregate the results, run results for TFA
    configs['_BASE_'] = base_cfg
    configs['DATASETS']['TRAIN'] = make_tuple(configs['DATASETS']['TRAIN'])
    configs['DATASETS']['TEST'] = make_tuple(configs['DATASETS']['TEST'])
    
    # if args.coco and not args.novel_finetune:
    #     # NOTE used for tfa cg (novel weights are different for different seed, but with same base weihgts)
    #     ckpt_path = os.path.join(output_dir, prefix, 'model_reset_combine.pth')
    #     # NOTE create the combined model if it doest not exist for the tfa coco (with cg)
    #     # for HDA, the model should be randomly initialized without the combination below
    #     if not os.path.exists(ckpt_path):
    #         src2 = os.path.join(
    #             output_dir, 'faster_rcnn_R_101_FPN_ft_novel_{}shot{}'.format(
    #                 shot, args.suffix),
    #             'model_final.pth',
    #         )
    #         if not os.path.exists(src2):
    #             print('Novel weights do not exist. Please run with the ' + \
    #                   '--novel-finetune flag first.')
    #             assert False
    #         combine_cmd = 'python -m tools.ckpt_surgery --coco --method ' + \
    #             'combine --src1 checkpoints/coco/faster_rcnn/faster_rcnn' + \
    #             '_R_101_FPN_base/model_final.pth --src2 {}'.format(src2) + \
    #             ' --save-dir {}'.format(os.path.join(output_dir, prefix))
    #         run_cmd(combine_cmd)
    #         assert os.path.exists(ckpt_path)
    #     configs['MODEL']['WEIGHTS'] = ckpt_path

    # elif not args.coco:
    if not args.coco:
        # NOTE to use randinit weight, to make the base1,2,3/model_reset_surgery.pth be prepared
        # for coco, only the 1 surgery model need to be prepared (tfa)
        # for HDA, prepare 1 surgery_ts (only 1 split)
        configs['MODEL']['WEIGHTS'] = configs['MODEL']['WEIGHTS'].replace(
            'base1', 'base' + str(args.split))
        configs['MODEL']['PRETRAINED_BASE_MODEL'] = configs['MODEL']['WEIGHTS'].replace(
            'base1', 'base' + str(args.split))
        for dset in ['TRAIN', 'TEST']:
            configs['DATASETS'][dset] = (
                configs['DATASETS'][dset][0].replace(
                    temp_mode, 'all' + str(args.split)),
            )
    # NOTE training set and test set are speficied by the split, training set is further specified by shot and seed   
    # TODO change the number of Newton iter for differet shots if needed (for 30 shots, test first to determine the number of iter)
    configs['DATASETS']['TRAIN'] = (
        configs['DATASETS']['TRAIN'][0].replace(
            '1shot', str(shot) + 'shot'
        ) + ('_{}'.format(seed_str) if seed_str != '' else ''),
    )
    configs['SOLVER']['BASE_LR'] = args.lr
    configs['SOLVER']['MAX_ITER'] = ITERS[shot][1]
    configs['SOLVER']['STEPS'] = (ITERS[shot][0],)
    # configs['CG_PARAMS']['NUM_CG_ITER'] = ITERS[shot][0]
    # configs['CG_PARAMS']['NUM_NEWTON_ITER'] = ITERS[shot][1]
    configs['SOLVER']['CHECKPOINT_PERIOD'] = ITERS[shot][1] // args.ckpt_freq
    configs['OUTPUT_DIR'] = os.path.join(output_dir, prefix)

    # if seed != 0:
    #     with open(save_file, 'w') as fp:
    #         yaml.dump(configs, fp)
    if (args.split==1 and shot ==1 and seed == 0) != True:
        with open(save_file, 'w') as fp:
            yaml.dump(configs, fp)

    return save_file, configs


def main(args):
    for shot in args.shots:
        for seed in range(args.seeds[0], args.seeds[1]):
            print('Split: {}, Seed: {}, Shot: {}'.format(args.split, seed, shot))
            cfg, configs = get_config(seed, shot)
            run_exp(cfg, configs, eval_saved_results = args.eval_saved_results)


if __name__ == '__main__':
    args = parse_args()
    main(args)
