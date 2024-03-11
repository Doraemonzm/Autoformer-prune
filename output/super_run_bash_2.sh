#!/bin/bash
# python -m torch.distributed.launch  --nproc_per_node=8  --use_env my_train.py --data-path /data/public/imagenet2012 --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./output/supernet_iter_2.yaml --prune ./output/super_1/checkpoint.pth --epochs 300 --epoch_intervel 100 --warmup_epochs 5 --output ./output/super_2 --batch-size 128 --lr 5e-05

python -m torch.distributed.launch  --nproc_per_node=8  --use_env my_train.py --data-path /home/pdl/datasets/ImageNet/ --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./output/supernet_iter_2.yaml --resume ./output/super_2/checkpoint.pth --epochs 300 --epoch_intervel 100 --warmup_epochs 5 --output ./output/super_2_resume --batch-size 128 --lr 5e-05
