import torch
import numpy as np
import MinkowskiEngine as ME
from DeepUrfold.Metrics.MetricSaver import MetricSaver, custom_metrics

class RawMetricSaver(MetricSaver):
    def __init__(self, hparams, dm, metrics=None, prefix=None, per_sample=False, save_labels=False):
        super().__init__(hparams, prefix=prefix, per_sample=per_sample)

        self.dm = dm
        self.save_labels = save_labels

        if metrics is None:
            metrics = ["raw"]
        elif not isinstance(metrics, (list, tuple)):
            metrics = [metrics]

        if self.save_labels:
            metrics += [f"labels-{m}" for m in metrics]

        for metric_type in metrics:
            self.metrics[metric_type] = custom_metrics.RawMetric(metric_type, compute_on_step=False)
            self.save_metric[metric_type] = [metric_type]
            print("Saved", metric_type)

    def on_test_batch_end(self, trainer, pl_module, result, batch, batch_idx, dataloader_idx):
        if isinstance(result, (list, tuple)):
            if len(result) != len(self.metrics)-len(self.use_other_metric):
                raise RuntimeError("The number of returned results must match the number of metrics that do not use other metrics and will be added in the same order. Total: {} ({}); Ignore: {} ({})".format(len(result), self.metrics, len(self.metrics)-len(self.use_other_metric), self.use_other_metric))

            device = result[0].device
            multiple_results = True
        else:
            device = result.device
            multiple_results = False

        if len(batch) == 2:
            coords, labels = batch
        else:
            coords, data, labels = batch
            other_labels = labels[:, data.size()[1]:].int()
            labels = labels[:, :data.size()[1]]


            if self.per_sample and other_labels.size()[1]==1:
                other_labels = torch.unique(other_labels[:,0])

            del data

        n_samples = len(torch.unique(coords[:, 0]))

        i = 0

        for metric_name, metric_func in self.metrics.items():
            if metric_name in self.use_other_metric: continue
            if "labels" in metric_name: continue

            if multiple_results:
                r = result[i]
            else:
                r = result

            if False and self.per_sample:
                if isinstance(r, ME.SparseTensor):
                    for sample_pred in r.decomposed_features:
                        metric_func(sample_pred)
                elif len(r)==n_samples:
                    for v in r:
                        metric_func(v)
                else:
                    assert 0, (len(r), n_samples, coords)

            else:
                if isinstance(r, ME.SparseTensor):
                    metric_func(r.F) #pred
                else:
                    metric_func(r)

            i += 1

        if self.save_labels:
            for metric_name, metric_func in self.metrics.items():
                if not metric_name.startswith("labels"): continue
                metric_func(other_labels)

    def post_process_value(self, metric_name, value_name, value):
        if metric_name.startswith("labels"):
            return np.array(self.dm.label_encoder.inverse_transform(value.cpu().numpy()))
        return value
