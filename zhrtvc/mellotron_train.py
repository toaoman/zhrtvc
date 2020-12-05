import sys
import os
import yaml

import torch

from mellotron.hparams import create_hparams
from mellotron.train import train, json_dump, parse_args

args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

if __name__ == '__main__':
    try:
        from setproctitle import setproctitle

        setproctitle('zhrtvc-mellotron-train')
    except ImportError:
        pass

    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    meta_folder = os.path.join(args.output_directory, 'metadata')
    os.makedirs(meta_folder, exist_ok=True)

    path = os.path.join(meta_folder, "args.json")
    obj = args.__dict__
    json_dump(obj, path)

    print('{}\nargs:'.format('-' * 50))
    print(yaml.dump(args.__dict__))

    print('{}\nhparams:'.format('-' * 50))
    print(yaml.dump({k: v for k, v in hparams.items()}))

    train(args.input_directory, args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
