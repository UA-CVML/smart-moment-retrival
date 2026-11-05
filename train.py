"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random
import datetime
from glob import glob

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import lavis_1019.tasks as tasks
from lavis_1019.common.config import Config
from lavis_1019.common.dist_utils import get_rank, init_distributed_mode
from lavis_1019.common.logger import setup_logger
from lavis_1019.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis_1019.common.registry import registry
from lavis_1019.common.utils import now

from lavis_1019.datasets.builders import *
from lavis_1019.models import *
from lavis_1019.processors import *
from lavis_1019.runners import *
from lavis_1019.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="configuration path relative to the repository root.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into the config file.",
    )

    args = parser.parse_args()

    return args

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls

def main():
    # Resolve all CLI paths from this checkout, regardless of the caller's directory.
    os.chdir(registry.get_path("repo_root"))

    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    setup_logger()

    if cfg.run_cfg.wandb and get_rank() == 0:
        wandb.init(
            project=cfg.run_cfg.wandb_project,
            name=cfg.run_cfg.wandb_name,
            config=cfg.to_dict(),
            job_type="train",
        )

    def count_directories(parent_directory, pattern):
        return len(
            [
                d
                for d in glob(
                    os.path.join(parent_directory, pattern)
                )
                if os.path.isdir(d)
            ]
        )

    num_dirs = count_directories(cfg.run_cfg.output_dir, f"{cfg.run_cfg.wandb_name}*")
    run_name = f"{cfg.run_cfg.wandb_name}-{num_dirs + 1}"

    cfg.run_cfg.output_dir = os.path.join(cfg.run_cfg.output_dir, run_name)

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg) 
    model = task.build_model(cfg)
    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()

if __name__ == "__main__":
    main()
