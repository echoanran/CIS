from sys import flags
import numpy as np
import os
import argparse
from ruamel import yaml
from torch.autograd.grad_mode import F
from au_lib.data_utils import compute_label_frequency

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--processor_name',
                        type=str,
                        default='train-image-causal',
                        help='processor name')
    parser.add_argument('-c',
                        '--config_dir',
                        type=str,
                        default='./config/exp1',
                        help='config dir name')
    parser.add_argument('-w',
                        '--work_dir',
                        type=str,
                        default='./work_dir/train/bp4d/exp1',
                        help='work dir name')
    parser.add_argument('-d',
                        '--data_dir',
                        type=str,
                        default='./data/bp4d_example',
                        help='data dir name')
    parser.add_argument('-k', '--kfold', type=int, default=3, help='kfold')
    parser.add_argument('--num_class',
                        type=int,
                        default=12,
                        help='num of class to detect')

    args = parser.parse_args()

    if not os.path.exists(args.config_dir):
        os.mkdir(args.config_dir)

    for k in range(args.kfold):

        label_freq = compute_label_frequency(
            os.path.join(args.data_dir, 'train' + str(k) + '_label.pkl'))

        desired_caps = {
            'work_dir': os.path.join(args.work_dir, str(k)),
            'feeder': 'feeder.feeder_image_causal.Feeder',
            'train_feeder_args': {
                'label_path':
                os.path.join(args.data_dir, 'train' + str(k) + '_label.pkl'),
                'image_path':
                os.path.join(args.data_dir,
                             'train' + str(k) + '_imagepath.pkl'),
                'image_size':
                256,
                'istrain':
                True,
            },
            'test_feeder_args': {
                'label_path':
                os.path.join(args.data_dir, 'val' + str(k) + '_label.pkl'),
                'image_path':
                os.path.join(args.data_dir, 'val' + str(k) + '_imagepath.pkl'),
                'image_size':
                256,
                'istrain':
                False,
            },
            'model': 'net.CISNet.Model',
            'model_args': {
                'num_class': args.num_class,
                'backbone': 'resnet34',
                'temporal_model': 'single',
                'subject': True,
                'pooling': True,
                'd_in': 512,
                'd_m': 256,
                'd_out': 512,
            },
            'log_interval': 1000,
            'save_interval': 5,
            'device': [0],
            'batch_size': 4,
            'test_batch_size': 4,
            'base_lr': 0.001,
            'lr_decay': 0.3,
            'step': [],
            'num_epoch': 15,
            'debug': False,
            'num_worker': 1,
            'optimizer': 'SGD',
            'weight_decay': 0.0005,
            'loss': 'clf',
            'loss_weight': label_freq.tolist(),
            'pretrain': True,
            'seed': 42,
        }

        yamlpath = os.path.join(args.config_dir, 'train' + str(k) + '.yaml')
        with open(yamlpath, "w", encoding="utf-8") as f:
            yaml.dump(desired_caps, f, Dumper=yaml.RoundTripDumper)

        cmdline = "python main.py " + args.processor_name + " -c " + yamlpath
        print(cmdline)
        os.system(cmdline)
