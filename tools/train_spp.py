import argparse
import torch
from core.config import cfg
from core.solver import *
from core.data import *
from core.model import *
from core.utils.general.registry_factory import SOLVER_REGISTRY
import multiprocessing
# multiprocessing.get_start_method()

def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train a retrieval network")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="config file", default=None, type=str)
    parser.add_argument(
        "--save_path", dest="save_path", help="config file", default='CARSruns', type=str)
    return parser.parse_args()


def main():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    arguments = dict()
    arguments["iteration"] = 0
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.TB_SAVE_DIR = f'{args.save_path}/events'
    cfg.SAVE_DIR = f'{args.save_path}/chekpoints'
    solver = SOLVER_REGISTRY[cfg.SOLVER.SOLVER_TYPE](cfg)
    with torch.autograd.set_detect_anomaly(True):
        solver.train()


if __name__ == '__main__':
    main()
