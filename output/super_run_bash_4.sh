#!/bin/bash
python -m torch.distributed.launch  --nproc_per_node=8  --use_env my_train.py --data-path /home/pdl/datasets/ImageNet/ --gp --change_qk --relative_position --mode retrain --dist-eval --cfg output/supernet_iter_4.yaml --prune ./output/super_3/checkpoint.pth --epochs 300 --warmup_epochs 5 --output ./output/super_4 --batch-size 128 --lr 5e-05
