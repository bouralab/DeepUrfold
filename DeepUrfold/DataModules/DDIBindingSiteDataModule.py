from DeepUrfold.DataModules.DomainStructureDataModule import DomainStructureDataModule
from DeepUrfold.Datasets.DDIBindingSiteDataset import DDIBindingSiteDataset

class DDIBindingSiteDataModule(DomainStructureDataModule):
    DATASET = DDIBindingSiteDataset

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = DomainStructureDataModule.add_data_specific_args(parent_parser)

        parser.set_defaults(nClasses="bool2")
        
        # parser.add_argument("--mask-binding-sites", type=str2bool, nargs='?',
        #                    const=True, default=False)
        return parser
