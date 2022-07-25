import os, sys

from DeepUrfold.Datasets.DomainStructureDataset import DomainStructureDataset
from DeepUrfold.Models.DomainStructureVAE import DomainStructureVAE
from DeepUrfold.Evaluators.Evaluator import DomainStructureEvaluator

class SuperfamilyVAE(DomainStructureEvaluator):
    DATASET = DomainStructureDataset
    MODEL = DomainStructureVAE
    LRP_VARS = {"mean":0}

    def lrp_model(self):
        return self.model.encoder


if __name__ ==  '__main__':
    trainer = SuperfamilyVAE()
    trainer.test()
