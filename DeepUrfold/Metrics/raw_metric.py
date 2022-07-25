# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Optional

import torch
from distutils.version import LooseVersion

from torchmetrics import Metric
from torchmetrics.functional.classification import auc, auroc
from pytorch_lightning.utilities import rank_zero_warn


class RawMetric(Metric):
    """
     Args:
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    """
    def __init__(
        self,
        metric_name: str = "raw",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.metric_name = metric_name

        self.add_state(metric_name, default=[], dist_reduce_fx=None)

    def update(self, data: torch.Tensor, *args, **kwds):
        """
        Update state with predictions and targets.

        Args:
            data: Predictions from model (probabilities, or labels)
        """
        getattr(self, self.metric_name).append(data)

    def compute(self) -> torch.Tensor:
        """
        Computes AUROC based on inputs passed in to ``update`` previously.
        """
        values = torch.cat(getattr(self, self.metric_name), dim=0)
        return values

        # return _auroc_compute(
        #     preds,
        #     target,
        #     self.mode,
        #     self.num_classes,
        #     self.pos_label,
        #     self.average,
        #     self.max_fpr
        # )
