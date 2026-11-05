"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis_1019.processors.base_processor import BaseProcessor
from lavis_1019.processors.blip_processors import (
    Blip2VideoTrainProcessor,
    BlipVideoEvalProcessor,
    BlipQuestionProcessor,
)
from lavis_1019.common.registry import registry

__all__ = ["BaseProcessor", "Blip2VideoTrainProcessor", "BlipVideoEvalProcessor", "BlipQuestionProcessor"]

def load_processor(name, cfg=None):
    return registry.get_processor_class(name).from_config(cfg)
