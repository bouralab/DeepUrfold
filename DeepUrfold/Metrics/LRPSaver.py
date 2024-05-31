import pandas as pd

#Load lrp investigator
from pytorch_lrp.innvestigator import InnvestigateModel

#Load MinkowskiEngine lrp modules (Must be before ME to fix SparseTensor importing)
from pytorch_lrp.minkowski_lrp import *

import MinkowskiEngine as ME
from DeepUrfold.Metrics.RawMetricSaver import RawMetricSaver

class LRPSaver(RawMetricSaver):
    def __init__(self, hparams, dm, all_metrics, lrp_metric, atomic_relevance, lrp_innvestigator=None,  prefix=None, per_sample=False):
        assert isinstance(lrp_metric, dict) and len(lrp_metric)==1, "lrp_metric must a be single with a single entry of {var_name:index_in_all_metrics}"
        super().__init__(hparams, dm, metrics=all_metrics, prefix=prefix, per_sample=per_sample)
        self.lrp_innvestigator = lrp_innvestigator
        self.atomic_relevance = atomic_relevance
        self.lrp_metric = lrp_metric
        self.checkpoint = self.hparams.checkpoint

    def set_model(self, model):
        self.lrp_innvestigator = InnvestigateModel(model)

    def on_test_batch_end(self, trainer, pl_module, result, batch, batch_idx, dataloader_idx):
        super().on_test_batch_end(trainer, pl_module, result, batch, batch_idx, dataloader_idx)
        self.run_lrp(result, batch, batch_idx)

    def run_lrp(self, result, batch, batch_idx):
        coords, data, labels = batch

        assert data.size()[1] < labels.size()[1]

        cath_domains = ME.SparseTensor(labels[:, data.size()[1]:].float(), coords.int())

        lrp_var = result[list(self.lrp_metric.values())[0]]
        _, relevances = self.lrp_innvestigator.innvestigate(lrp_var, no_recalc=True, autoencoder_in=True, rule="LayerNumRule")

        assert torch.all(torch.eq(coords.cpu(), relevances.C.cpu())), f"Coords does not equal relevance coords: {coords} {relevances.C}"

        num_batches = int(np.ceil(len(self.dm.test_dataset)/self.hparams.batch_size))
        num_relevances = len(relevances.decomposed_features)

        if batch_idx < num_batches-1:
            assert num_relevances == self.hparams.batch_size, f"Number of relevances {num_relevances} does not equal the batch size ({self.hparams.batch_size}) {result[0].size()} {labels.size()}"
        else:
            last_batch_size = len(self.dm.test_dataset)-self.hparams.batch_size*batch_idx
            #print(f"Number of relevances {num_relevances} in the last batch does not equal the correct batch size ({last_batch_size})", len(self.dm.test_dataset.order), self.hparams.batch_size, batch_idx)
            assert num_relevances == last_batch_size, f"Number of relevances {num_relevances} in the last batch does not equal the correct batch size ({last_batch_size})"

        for sample_idx, (coords, relevance) in enumerate(zip(*relevances.decomposed_coordinates_and_features)):
            current_idx = self.hparams.batch_size*batch_idx+sample_idx

            cath_domain_embedding = cath_domains.features_at(sample_idx)[0]
            cath_domain = self.dm.label_encoder.inverse_transform(cath_domain_embedding.cpu().int().numpy())[0]
            superfamily  = self.dm.domain_to_superfamily[cath_domain]

            index = pd.MultiIndex.from_tuples([tuple(c) for c in coords.cpu().int().numpy()], names=["x", "y", "z"])
            relevance = pd.DataFrame(relevance.cpu().numpy(), index=index)
            
            self.atomic_relevance.update(relevance, cath_domain, superfamily, self.checkpoint, batch_index=batch_idx)
