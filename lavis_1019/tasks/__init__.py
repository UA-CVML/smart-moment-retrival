"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis_1019.common.registry import registry
from lavis_1019.tasks.base_task import BaseTask
from lavis_1019.tasks.moment_retrieval import MomentRetrievalTask

def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task # moment_retrieval
    # get_task_class:lavis_1019.tasks.moment_retrieval.MomentRetrievalTask
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = ["BaseTask", "MomentRetrievalTask", "setup_task"]
