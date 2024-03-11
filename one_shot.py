import random
import os
import subprocess
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

from lib.datasets import build_dataset
from lib import utils
from supernet_engine import evaluate
from model.supernet_transformer import Vision_TransformerSuper
import argparse

import yaml
from lib.config import cfg, update_config_from_file
from helper import build_yaml

Train_CFG = {
    'data-path': None, 'gp': True, 'change_qk': True,  'relative_position': True, 'mode': 'retrain', 'dist-eval': True, 'cfg': None,'prune': None,
    'epochs': None, 'epoch_intervel': None, 'warmup_epochs':5, 'output': None, 'batch-size': 112, 'resume': None, 'lr': None
}

Search_CFG = {
    'data-path': None, 'gp': True, 'change_qk': True,  'relative_position': True, 'dist-eval': True, 'cfg': None,  'resume': None,
    'min-param-limits': None, 'param-limits': None, 'data-set': 'IMNET'
}



def get_args_parser():
    parser = argparse.ArgumentParser('my script', add_help=False)

    # mainstream
    parser.add_argument('--epoch_intervel', default=50, type=int)
    # parser.add_argument('--mannual_intervel', default= [50, 150, 150], type=int)
    parser.add_argument('--prune_num', default=2, help='total num for prune', type=int)
    parser.add_argument('--gpu_num', default=8, help='total num for prune', type=int)
    parser.add_argument('--resume_iter', default=None, help='resume search iter', type=int)
    # parser.add_argument('--max_param', type=float, default=23)
    # parser.add_argument('--min_param', type=float, default=18)

    # for supernet
    parser.add_argument('--super_train_data', default='/home/pdl/datasets/ImageNet/', type=str,
                        help='dataset path')


    parser.add_argument('--super_epoch', default=300, help='total epochs for supernet train', type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--super_save', default='./output/super', help='path where to save supernet output ')
    parser.add_argument('--mode', default=None, help='supernet mode, retrain or super ')




    # for subnet search

    parser.add_argument('--search_data_path', default='/home/pdl/datasets/ImageNet/', type=str,
                        help='dataset path')

    parser.add_argument('--search_resume', default='', help='resume from checkpoint')

    parser.add_argument('--search_save', default='./output/search', help='path where to save supernet output ')


    return parser




def exec_supernet_train(xargs, idx, super_cfg_path=None, is_prune=True):
    bash_file = ['#!/bin/bash']
    execution_line = "python -m torch.distributed.launch  --nproc_per_node={}  --use_env my_train.py".format(xargs.gpu_num)
    Train_CFG['data-path']=xargs.super_train_data
    Train_CFG['cfg'] = super_cfg_path
    Train_CFG['epochs'] = xargs.super_epoch
    Train_CFG['warmup_epochs'] = xargs.warmup_epochs
    Train_CFG['lr'] = xargs.lr
    Train_CFG['batch-size'] = xargs.batch_size
    Train_CFG['output'] = xargs.super_save +'_'+str(idx)

    if is_prune:
        Train_CFG['epoch_intervel'] = xargs.epoch_intervel
        # if xargs.mannual_intervel:
        #     Train_CFG['epoch_intervel'] = xargs.mannual_intervel[idx]
        # else:
        #     Train_CFG['epoch_intervel'] = xargs.epoch_intervel
    else:
        Train_CFG['epoch_intervel'] = None
        # Train_CFG['mode'] = 'retrain'
    if idx>0:
        Train_CFG['prune'] = os.path.join(xargs.super_save +'_'+str(idx-1), 'checkpoint.pth')

    if xargs.mode:
        Train_CFG['mode'] = xargs.mode

    for k, v in Train_CFG.items():
        if v is not None:
            if isinstance(v, bool):
                if v:
                    execution_line += " --{}".format(k)
            else:
                execution_line += " --{} {}".format(k, v)
    # execution_line += ' &'
    bash_file.append(execution_line)
    # bash_file.append('wait')
    # save=

    with open('output/super_run_bash_{}.sh'.format(idx), 'w') as handle:
        for line in bash_file:
            handle.write(line + os.linesep)
    subprocess.call("sh output/super_run_bash_{}.sh".format(idx), shell=True)





def exec_subnet_search(xargs, idx, super_cfg_path=None, super_resume=None):
    bash_file = ['#!/bin/bash']
    execution_line = "python -m torch.distributed.launch  --nproc_per_node={}  --use_env my_search.py".format(xargs.gpu_num)
    Search_CFG['data-path']=xargs.search_data_path
    Search_CFG['cfg'] = super_cfg_path
    Search_CFG['resume'] = super_resume
    Search_CFG['min-param-limits'] = 25 - 6*idx
    Search_CFG['param-limits'] = 34 - 6*idx
    Search_CFG['output_dir']= xargs.search_save+'_'+str(idx)
    # if pruned_cfg_path:
    #     Search_CFG['pruned_cfg'] = pruned_cfg_path

    for k, v in Search_CFG.items():
        if v is not None:
            if isinstance(v, bool):
                if v:
                    execution_line += " --{}".format(k)
            else:
                execution_line += " --{} {}".format(k, v)
    # execution_line += ' &'
    bash_file.append(execution_line)
    # bash_file.append('wait')


    with open('output/search_run_bash_{}.sh'.format(idx), 'w') as handle:
        for line in bash_file:
            handle.write(line + os.linesep)
    subprocess.call("sh output/search_run_bash_{}.sh".format(idx), shell=True)









def main(args):
    for idx in range(args.prune_num):

        super_dir= args.super_save + '_' + str(idx)
        search_dir= args.search_save + '_' + str(idx)

        if not os.path.exists(super_dir):
            os.makedirs(super_dir, exist_ok=True)
        if not os.path.exists(search_dir):
            os.makedirs(search_dir, exist_ok=True)

        if args.resume_iter and idx< args.resume_iter:
            continue

        super_cfg= 'output/supernet_iter_{}.yaml'.format(idx)

        exec_supernet_train(args, idx, super_cfg_path=super_cfg)

        exec_subnet_search(args, idx, super_cfg_path=super_cfg, super_resume=os.path.join(super_dir, 'checkpoint.pth'))

        load_from_search = \
            torch.load(os.path.join(search_dir, 'search_checkpoint.pth.tar'))['keep_top_k'][1][0]

        build_yaml(load_from_search, idx)

    final_cfg = os.path.join('output', 'supernet_iter_{}.yaml'.format(args.prune_num))

    exec_supernet_train(args, idx=args.prune_num, super_cfg_path=final_cfg,  is_prune=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MyAutoFormer', parents=[get_args_parser()])
    args = parser.parse_args()
    Path('output').mkdir(parents=True, exist_ok=True)
    main(args)