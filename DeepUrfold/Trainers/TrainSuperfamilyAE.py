import os, sys
sys.path.append("/project/ppi_workspace/toil-env36/lib/python3.6/site-packages/")

from DeepUrfold.Datasets.DomainStructureDataset import DomainStructureDataset
from DeepUrfold.Models.SuperfamilyAE import SuperfamilyAE
from DeepUrfold.Trainers.Train import DomainStructureTrainer

class SuperfamilyAE(DomainStructureTrainer):
    DATASET_CLS = DomainStructureDataset
    MODEL_CLS = SuperfamilyAE

if __name__ ==  '__main__':
    trainer = SuperfamilyAE()
    trainer.fit()
