from pathlib import Path
import torch
import pickle
import argparse
import logging
import torch.distributed as dist
from config import MyParser
from steps import trainer


if __name__ == "__main__":
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    torch.cuda.empty_cache()
    args = MyParser().parse_args()
    logging.info(args)
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"exp_dir: {str(exp_dir)}")

    if args.resume:
        resume = args.resume
        assert(bool(args.exp_dir))
        with open("%s/args.pkl" % args.exp_dir, "rb") as f:
            old_args = pickle.load(f)
        new_args = vars(args)
        old_args = vars(old_args)
        for key in new_args:
            if key not in old_args or old_args[key] != new_args[key]:
                old_args[key] = new_args[key]
        args = argparse.Namespace(**old_args)
        args.resume = resume
    else:
        with open("%s/args.pkl" % args.exp_dir, "wb") as f:
            pickle.dump(args, f)

    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    my_trainer = trainer.Trainer(args, world_size, rank)
    my_trainer.train()