from DeepUrfold.Models.DomainStructureSegmentationModel import DomainStructureSegmentationModel

class SuperfamilyAE(DomainStructureSegmentationModel):
    def prepare_data(self, prefix=None):
        prefix = prefix if prefix is None else "DomainStructureDataset"
        super().prepare_data(prefix=prefix)
