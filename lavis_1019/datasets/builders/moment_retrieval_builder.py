"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis_1019.common.registry import registry
from lavis_1019.common.utils import get_cache_path
from lavis_1019.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis_1019.datasets.datasets.moment_retrieval_dataset import MomentRetrievalDataset


class MomentRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = MomentRetrievalDataset
    eval_dataset_cls = MomentRetrievalDataset

    def build(self):
        datasets = super().build()

        return datasets


@registry.register_builder("qvh")
class QVHBuilder(MomentRetrievalBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/qvh/defaults.yaml",
    }
