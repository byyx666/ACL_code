import json
import argparse
from trainer import train

default_config = './exps/l2p_inr.json'

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    
    if args['gpu_'] is not None:
        args['device'] = [str(args['gpu_'])]
    if args['dataset_']:
        args['dataset'] = args['dataset_']
    if args['inc_']:
        args['init_cls'] = args['inc_']
        args["increment"] = args['inc_']
    if args['warm_'] is not None:
        args['warm'] = args['warm_']

    file_id = "{}-{}-inc{}".format(args["model_name"],args["dataset"],str(args['increment']))

    if "warm" in args and args["warm"] is True:
        file_id = file_id + '-warm'
    file_id = file_id + ''

    train(args, file_id)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default=default_config,
                        help='Json file of settings.')
    parser.add_argument('--gpu_', type=int, default=None,
                        help='ID of used gpu, the default value is defined in the config file.')
    parser.add_argument('--dataset_', type=str, default=None,
                        help='Name of dataset, the default value is defined in the config file.')
    parser.add_argument('--inc_', type=int, default=None,
                        help='')
    parser.add_argument('--warm_', action='store_true', 
                        help='')
    
    return parser

if __name__ == '__main__':
    main()
