from torch import nn
import torch
import torch.nn.functional as F

#from pytorch_lightning.metrics import F1, AUROC, AveragePrecision, DiceCoefficient, IoU
#from DeepUrfold.Models.metric import TensorMetric
# from pytorch_lightning.metrics import classification
# classification.TensorMetric = TensorMetric
# F1 = classification.F1
# AUROC = classification.AUROC
# AveragePrecision = classification.AveragePrecision
# DiceCoefficient = classification.DiceCoefficient
# IoU = classification.IoU

from DeepUrfold.Models.DomainStructureSegmentationModel import DomainStructureSegmentationModel, str2bool
from DeepUrfold.Datasets.DDIBindingSiteDataset import DDIBindingSiteDataset

class BindingSitePredictor(DomainStructureSegmentationModel):
    def __init__(self, hparams):
        metrics = {
            #(metric, use_bool)
            # "F1": (F1(), True),
            # "AUROC": (AUROC(), True),
            # "AUPRC": (AveragePrecision(), True),
            # "Dice": (DiceCoefficient(), False),
            # "IoU": (IoU(), True)
        }
        super().__init__(hparams, metrics)
        self.activation = nn.Softmax(dim=1)


    def loss(self, outputs, labels, bfactors=None):
        return F.cross_entropy(outputs, torch.max(labels, 1)[1])

    def forward(self, inputs):
        outputs = super().forward(inputs)
        outputs = self.activation(outputs)
        return outputs

    @staticmethod
    def add_model_specific_args(parent_parser):
        return DomainStructureSegmentationModel.add_model_specific_args(parent_parser)
