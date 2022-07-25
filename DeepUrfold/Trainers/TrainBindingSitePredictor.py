import os, sys
sys.path.append("/project/ppi_workspace/py37/lib/python3.7/site-packages")

#os.environ["LD_LIBRARY_PATH"] = "/torch/build/lib:{}".format(os.environ["LD_LIBRARY_PATH"])

from DeepUrfold.DataModules.DDIBindingSiteDataModule import DDIBindingSiteDataModule
from DeepUrfold.Models.BindingSitePredictor import BindingSitePredictor
from DeepUrfold.Trainers.Train import DomainStructureTrainer

class DDIBindingSiteTrainer(DomainStructureTrainer):
    DATAMODULE = DDIBindingSiteDataModule
    MODEL = BindingSitePredictor

if __name__ ==  '__main__':
    trainer = DDIBindingSiteTrainer()
    trainer.fit()
