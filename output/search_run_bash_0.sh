#!/bin/bash
python -m torch.distributed.launch  --nproc_per_node=8  --use_env my_search.py --data-path subImageNet --gp --change_qk --relative_position --dist-eval --cfg ./output/supernet_iter_1.yaml --resume ./output/super_1/checkpoint.pth --min-param-limits 45 --param-limits 65 --data-set EVO_IMNET --output_dir ./output/search_1
