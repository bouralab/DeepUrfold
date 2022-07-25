import os, sys

from DeepUrfold.Models.DomainStructureVAE import DomainStructureVAE
from DeepUrfold.Trainers.Train import DomainStructureTrainer

class SuperfamilyVAE(DomainStructureTrainer):
    MODEL = DomainStructureVAE

if __name__ ==  '__main__':
    trainer = SuperfamilyVAE()
    trainer.fit()
