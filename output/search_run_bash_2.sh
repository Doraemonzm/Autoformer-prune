#!/bin/bash
python -m torch.distributed.launch  --nproc_per_node=8  --use_env my_search.py --data-path /home/pdl/datasets/ImageNet/ --gp --change_qk --relative_position --dist-eval --cfg ./output/supernet_iter_2.yaml --resume ./output/super_2/checkpoint.pth --min-param-limits 25 --param-limits 45 --data-set EVO_IMNET --output_dir ./output/search_2
