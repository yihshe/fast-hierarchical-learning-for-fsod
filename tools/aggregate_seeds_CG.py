from IPython.terminal.embed import embed
import numpy as np
import matplotlib.pyplot as plt

import argparse
import json
import math
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shots', type=int, default=1,
                        help='Shots to aggregate over')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Seeds to aggregate over')
    # Model
    parser.add_argument('--fc', action='store_true',
                        help='Model uses FC instead of cosine')
    parser.add_argument('--unfreeze', action='store_true',
                        help='Unfreeze feature extractor')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of path')
    # Output arguments
    parser.add_argument('--print', action='store_true', help='Clean output')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    parser.add_argument('--save-dir', type=str, default='.',
                        help='Save dir for generated plots')
    # PASCAL arguments
    parser.add_argument('--split', type=int, default=1, help='Data split')
    # COCO arguments
    parser.add_argument('--coco', action='store_true', help='Use COCO dataset')

    # ckpt path
    parser.add_argument('--ckpt-path-prefix', type = str, default='checkpoints')
    # new setting or not
    parser.add_argument('--new-setting', action='store_true', help='Aggregate seeds for the results of new setting')

    args = parser.parse_args()
    return args


def main(args):
    metrics = {}
    num_ckpts = 0
    dataset = 'coco' if args.coco else 'voc'
    if args.fc:
        fc = '_fc'
    else:
        fc = '_normalized' if not args.coco else ''
    if args.unfreeze:
        unfreeze = '_unfreeze'
    else:
        unfreeze = '_randnovel' if not args.coco else ''
    # TODO aggregate seeds for a specific shot
    new_setting = '_new_setting' if args.new_setting else ''
    for i in range(args.seeds):
        seed = 'seed{}/'.format(i) if i != 0 else ''
        # TODO change the prefix of the path here to the specific TFA and HDA
        # prefix = 'checkpoints/{}/faster_rcnn/{}'.format(dataset, seed)
        prefix = '{}/{}/faster_rcnn/{}'.format(args.ckpt_path_prefix, dataset, seed)
        prefix += 'faster_rcnn_R_101_FPN_ft{}_all'.format(fc)
        if args.coco:
            ckpt = prefix + '_{}shot{}'.format(args.shots, unfreeze)
        else:
            ckpt = prefix + '{}_{}shot{}{}'.format(
                args.split, args.shots, unfreeze, args.suffix)
        # TODO add new setting here
        if os.path.exists(ckpt):
            if os.path.exists(os.path.join(ckpt, 'inference/all_res{}.json'.format(new_setting))):
                ckpt_ = os.path.join(ckpt, 'inference/all_res{}.json'.format(new_setting))
                res = json.load(open(ckpt_, 'r'))
                res = res[os.path.join(ckpt, 'model_final.pth')]['bbox']
            elif os.path.exists(os.path.join(ckpt, 'inference/res_final{}.json'.format(new_setting))):
                ckpt = os.path.join(ckpt, 'inference/res_final{}.json'.format(new_setting))
                res = json.load(open(ckpt, 'r'))['bbox']
            else:
                print('Missing: {}'.format(ckpt))
                continue
            
            for metric in res:
                if metric in metrics:
                    metrics[metric].append(res[metric])
                else:
                    metrics[metric] = [res[metric]]
            num_ckpts += 1
        else:
            print('Missing: {}'.format(ckpt))
    # print('Num ckpts: {}'.format(num_ckpts))
    # print('')

    # TODO new setting or not
    save_dir = '{}/{}/faster_rcnn/aggregated_results{}'.format(args.ckpt_path_prefix, dataset, new_setting)
    
    metrics_to_count = ['AP', 'AP50', 'AP75', 'bAP', 'bAP50', 'bAP75', 'aAP', 'aAP50', 'aAP75', 'fAP', 'fAP50', 'fAP75', 'nAP', 'nAP50', 'nAP75']

    
    # Output results
    if args.print:
        # Clean output for copy and pasting
        # NOTE print a list of AP names to statistic
        out_str = ''
        for metric in metrics:
            if metric in metrics_to_count:
                out_str += '&{} '.format(metric)
        if args.shots == 1:
            print(out_str)
        
        # NOTE print mean of APs
        means = []
        for metric in metrics:
            if metric in metrics_to_count:
                means.append('{0:.1f}'.format(np.mean(metrics[metric])))
        # print(means)
        
        # NOTE print CI of APs
        cis = []
        for metric in metrics:
            if metric in metrics_to_count:
                cis.append('{0:.1f}'.format(
                    1.96*np.std(metrics[metric]) / math.sqrt(len(metrics[metric]))
                ))
        # print(cis)
        
        output_row = ""
        for mean, ci in zip(means, cis):
            unit = "&{}$\pm${} ".format(mean, ci)
            output_row+=unit
        output_row += " \\\\"
        print(output_row)

        """
        # Clean output for copy and pasting
        # NOTE print a list of AP names to statistic
        out_str = ''
        for metric in metrics:
            if metric in metrics_to_count:
                out_str += '{} '.format(metric)
        print(out_str)
        
        # NOTE print mean of APs
        out_str = ''
        for metric in metrics:
            if metric in metrics_to_count:
                out_str += '{0:.1f} '.format(np.mean(metrics[metric]))
        print(out_str)
        
        # NOTE print CI of APs
        out_str = ''
        for metric in metrics:
            if metric in metrics_to_count:
                out_str += '{0:.1f} '.format(
                    1.96*np.std(metrics[metric]) / math.sqrt(len(metrics[metric]))
                )
        print(out_str)
        
        # NOTE print std of APs
        out_str = ''
        for metric in metrics:
            if metric in metrics_to_count:
                out_str += '{0:.1f} '.format(np.std(metrics[metric]))
        print(out_str)
        
        # NOTE print p25 of APs
        out_str = ''
        for metric in metrics:
            if metric in metrics_to_count:
                out_str += '{0:.1f} '.format(np.percentile(metrics[metric], 25))
        print(out_str)
        
        # NOTE print p50 of APs
        out_str = ''
        for metric in metrics:
            if metric in metrics_to_count:
                out_str += '{0:.1f} '.format(np.percentile(metrics[metric], 50))
        print(out_str)
        
        # NOTE print p75 of APs
        out_str = ''
        for metric in metrics:
            if metric in metrics_to_count:
                out_str += '{0:.1f} '.format(np.percentile(metrics[metric], 75))
        print(out_str)
        """
        
    else:
        # Verbose output
        res = {}
        for metric in metrics:
            print(metric)
            print('Mean \t {0:.4f}'.format(np.mean(metrics[metric])))
            print('CI \t {0:.4f}'.format(
                1.96*np.std(metrics[metric]) / math.sqrt(len(metrics[metric]))
                ))
            print('Std \t {0:.4f}'.format(np.std(metrics[metric])))
            print('Q1 \t {0:.4f}'.format(np.percentile(metrics[metric], 25)))
            print('Median \t {0:.4f}'.format(np.percentile(metrics[metric], 50)))
            print('Q3 \t {0:.4f}'.format(np.percentile(metrics[metric], 75)))
            print('')

            res[metric] = {}
            res[metric]['Mean'] = '{0:.2f}'.format(np.mean(metrics[metric]))
            res[metric]['CI'] = '{0:.2f}'.format(
                1.96*np.std(metrics[metric]) / math.sqrt(len(metrics[metric]))
            )
            res[metric]['Std'] = '{0:.2f}'.format(np.std(metrics[metric]))
            res[metric]['Q1'] = '{0:.2f}'.format(np.percentile(metrics[metric], 25))
            res[metric]['Median'] = '{0:.2f}'.format(np.percentile(metrics[metric], 50))
            res[metric]['Q3'] = '{0:.2f}'.format(np.percentile(metrics[metric], 75))
        
        os.makedirs(
            save_dir, exist_ok=True
        )
        with open(
            os.path.join(save_dir, 
            'inference_split{}_{}shots_vs_{}seeds.json'.format(args.split, args.shots, args.seeds)),
            "w",
        ) as fp:
            json.dump(res, fp, indent=2)

    # Plot results
    if args.plot:
        # os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        # TODO also plot the AP for animal and food (aAP and fAP) in the new split
        # we can also save the results and plot the mean with confidence interval range later on
        for met in ['avg', 'stdev', 'ci']:
            for metric, c in zip(['nAP', 'nAP50', 'nAP75'],
                                 ['bo-', 'ro-', 'go-']):
                if met == 'avg':
                    res = [np.mean(metrics[metric][:i+1]) \
                            for i in range(len(metrics[metric]))]
                elif met == 'stdev':
                    res = [np.std(metrics[metric][:i]) \
                            for i in range(1, len(metrics[metric])+1)]
                elif met == 'ci':
                    res = [1.96*np.std(metrics[metric][:i+1]) / \
                            math.sqrt(len(metrics[metric][:i+1])) \
                                for i in range(len(metrics[metric]))]
                plt.plot(range(1, len(metrics[metric])+1), res, c)
            plt.legend(['nAP', 'nAP50', 'nAP75'])
            plt.title('Split {}, {} Shots - Cumulative {} over {} Seeds'.format(
                args.split, args.shots, met.upper(), args.seeds))
            plt.xlabel('Number of seeds')
            plt.ylabel('Cumulative {}'.format(met.upper()))
            plt.savefig(os.path.join(
                # args.save_dir,
                save_dir,
                'split{}_{}shots_{}_vs_{}seeds.png'.format(
                    args.split, args.shots, met, args.seeds),
            ))
            plt.clf()


if __name__ == '__main__':
    args = parse_args()
    main(args)
