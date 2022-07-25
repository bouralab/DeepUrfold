import os
import re
import pickle
import argparse
import itertools as it
from functools import partial
from collections import OrderedDict

import torch

import MinkowskiEngine as ME
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interp
from matplotlib import rcParams
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn import metrics as skmetrics
from pytorch_lightning.callbacks import Callback
#from torchmetrics.functional.classification import auroc, multiclass_auroc
from torchmetrics import classification as classification_metrics
from DeepUrfold import Metrics as custom_metrics

from pytorch_lightning.utilities import rank_zero_only

def get_metrics_from_module(module):
    return {var: getattr(module, var) for var in dir(module) if var not in globals() and \
        not var.startswith("__")}

allowable_metrics = get_metrics_from_module(classification_metrics)
allowable_metrics.update(get_metrics_from_module(custom_metrics))

def cat(t, dim=0, ravel=False):
    if isinstance(t, (list, tuple)):
        print("CAT", t)
        result = torch.cat(t, dim=0)
    else:
        result = t

    if ravel:
        return result.view(-1)
    return result

def cat_ravel(t):
    return cat(t, dim=0).view(-1)

def clamp_cat(t, ravel=False):
    return torch.clamp(cat(t, ravel=ravel), min=0, max=1.)

def clamp_cat_ravel(t, ravel=False):
    return torch.clamp(cat_ravel(t), min=0, max=1.)

def one_hot_cat(t, num_classes=None, ravel=False):
    print("NUM CLASSES", num_classes)
    tt = cat(t, dim=0).long()
    print(tt)

    result = F.one_hot(tt, num_classes=num_classes)

    if ravel:
        return result.view(-1)

    return result

def one_hot_cat_ravel(t, num_classes=None):
    return one_hot_cat(t, num_classes=num_classes).view(-1)

def max_one_hot_cat_ravel(t):
    _, pred = torch.max(t, dim=1)
    t = torch.zeros_like(t)
    t[pred] = 1
    return t

class ELBOSaver(Callback):
    def __init__(self, hparams, prefix="test"):
        self.dm = dm
        self.data = None
        self.score_file = f"{prefix}-elbo_scores.csv"

    def on_test_batch_end(self, trainer, pl_module, elbo, batch, batch_idx, dataloader_idx):
        if self.data is None:
            self.data = self.dm.test_dataset.data.reset_index().assign(ELBO=np.nan)
        self.data.loc[batch_idx, "ELBO"] = elbo.item()

    def on_test_epoch_end(self, trainer, pl_module):
        self.data.to_csv(self.score_file)

class MetricSaver(Callback):
    def __init__(self, hparams, prefix=None, per_sample=False):
        self.hparams = hparams
        self.prefix = prefix if prefix is not None else self.hparams.prefix
        self.metrics = OrderedDict()
        self.use_other_metric = OrderedDict()
        self.save_metric = OrderedDict()
        self.metric_values = OrderedDict()
        self.saved_metric_files = OrderedDict()
        self.per_sample = per_sample

        if "feature_groups" in self.hparams and self.hparams.feature_groups is not None \
          and "original_features" in self.hparams and self.hparams.original_features is not None:
            if len(self.hparams.feature_groups) == 1:
                self.hparams.feature_groups = self.hparams.feature_groups[0].split(";")

            group_feats = [len(f.split(";")) for f in hparams.original_features.split("__")]
            assert len(group_feats)==len(hparams.feature_groups), (len(group_feats), len(hparams.feature_groups))
            self.group_feats_start = [0]+np.cumsum(group_feats).tolist()
        else:
            self.group_feats_start = None

    def on_test_batch_end(self, trainer, pl_module, result, batch, batch_idx, dataloader_idx):
        raise NotImplementedError

    @rank_zero_only
    def _copy_metrics(self, save=False):
        for metric_name, metric_rule in self.use_other_metric.items():
            if save and not metric_name in self.save_metric:
                continue
            for other_metric, other_metric_rules in  metric_rule.items():
                print(metric_name, other_metric_rules)

                try:
                    self.metrics[other_metric]._sync_dist()
                except AssertionError:
                    pass
                for other_value_name, func in other_metric_rules:
                    print("DEVICE", other_value_name, [t.device for t in getattr(self.metrics[other_metric], other_value_name)])
                new_values = [func(getattr(self.metrics[other_metric], other_value_name)).cpu() \
                    for other_value_name, func in other_metric_rules]
                print(new_values)
                print([v.shape for v in new_values])
                self.metrics[metric_name](*new_values)

    @rank_zero_only
    def process(self, trainer=None, pl_module=None, stage=None, save=True, no_compute=False): #on_test_epoch_end
        if save:
            print("SAVE")
            self._copy_metrics(save=True)
            for metric_name, value_names in self.save_metric.items():
                print(metric_name)
                for value_name in value_names:
                    value = getattr(self.metrics[metric_name], value_name)

                    if isinstance(value, (list, tuple)):
                        value = torch.cat(value, dim=0)

                    if hasattr(self, "post_process_value") and callable(self.post_process_value):
                        value = self.post_process_value(metric_name, value_name, value)

                    metric_save_path = f"{self.prefix}-{metric_name}-{value_name}"

                    if isinstance(value, torch.Tensor):
                        with open(metric_save_path+".pt", "wb") as f:
                            torch.save(value.cpu(), f)
                    elif isinstance(value, np.ndarray):
                        np.save(metric_save_path+".npy", value)
                    else:
                        raise RuntimeError("value must be torch.Tensor or numpy.array")
                    print("Saved to", metric_save_path)
                    try:
                        self.saved_metric_files[metric_name][value_name] = metric_save_path
                    except KeyError:
                        self.saved_metric_files[metric_name] = {value_name: metric_save_path}

                    if isinstance(value, torch.Tensor):
                        value.cuda()

        if no_compute or self.group_feats_start is not None:
            return

        self._copy_metrics()

        self.metric_values = {}
        for metric_name, metric in self.metrics.items():
            if "multi" in metric_name: continue
            metric._reset = metric.reset
            metric.reset = lambda: None
            for val_name in metric._defaults.keys():
                print("    old", val_name, getattr(metric, val_name))
            value = metric.compute()


            self.metric_values[metric_name] = value

            metric.reset = metric._reset

        self.process_epoch(**self.metric_values)

    def process_batch(self, result, batch, batch_idx):
        pass

    def process_samples(self, metric_func, result, truth):
        for sample_pred, sample_truth in zip(result.decomposed_features, truth.decomposed_features):
            metric_func(sample_pred, sample_truth)

    def process_epoch(self, *args, **kwds):
        pass

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()





class MetricSaverOld(Callback):
    def __init__(self, hparams, data_module, metric="ELBO", prefix="test", per_sample=False):
        self.hparams = hparams
        self.data_module = data_module
        self.data = None
        self.metric_name = metric
        self.per_sample = per_sample
        self.num_classes = 1
        self.device = None

        try:
            self.metric_func = allowable_metrics[metric]
            self.metric = self.metric_func(compute_on_step=per_sample)
        except KeyError:
            self.metric_func = None

        self.score_file = f"{prefix}-{metric}_scores.csv"

    def on_test_batch_end(self, trainer, pl_module, metric, batch, batch_idx, dataloader_idx):
        if self.metric_func is not None:
            coords, _, labels = batch
            truth = ME.SparseTensor(labels.float(), coords.int()).to(metric.device)

        if self.device is None:
            self.device = str(metric.device)

        for sample_idx, (sample_pred, sample_truth) in enumerate(zip(metric.decomposed_features, truth.decomposed_features)):
            value_names = None
            if self.metric_func is not None:
                #Define true if value is greater than 0.7
                #sample_pred = (sample_pred>0.7).float()
                if sample_pred.size()[1]:
                    self.num_classes = sample_pred.size()[1]
                    _, sample_pred = torch.max(sample_pred, axis=1)
                    _, sample_truth = torch.max(sample_truth, axis=1)

                else:
                    sample_pred = (sample_pred>=0.6).float()

                if self.per_sample:
                    #Calculate metric for this one sample
                    m = self.metric_func(num_classes=sample_pred.size()[1])
                    #Add extra dimension: eg: [0, 1] => [[0], [1]]
                    #sample_pred = torch.unsqueeze(sample_pred.float().flatten(), 1)
                    m.update(sample_pred, sample_truth)
                    sample_pred = m.compute().detach().cpu()
                    value_names = [f"{self.metric_name}_{i}" for i in range(sample_pred.size()[0])]
                else:
                    #Save all predictions from all samples, will calculate metric at end
                    value_names = ["predicted", "truth"]#[f"pred_{i}" for i in range(sample_pred.size()[0])]
                    #value_names += [f"true_{i}" for i in range(sample_truth.size()[0])]
                    sample_pred = sample_pred.detach().cpu().numpy().tolist(),
                    sample_truth = sample_truth.detach().cpu().numpy().tolist() #torch.cat((sample_pred, sample_truth), axis=0)
                    if self.data is None:
                        self.data = [[sample_pred], [sample_truth]]
                    else:
                        self.data[0].append(sample_pred)
                        self.data[1].append(sample_truth)

                    continue


            if value_names is None:
                metric_size = sample_pred.flatten().size()[0]
                value_names = [f"{self.metric_name}_{i}" for i in range(metric_size)] \
                    if metric_size > 1 else [self.metric_name]

            if self.data is None:
                self.data = self.dm.test_dataset.data.reset_index().assign(\
                    **{n:np.nan for n in value_names}).copy()
                # if metric_size == 1:
                #     self.data = self.dm.test_dataset.data.reset_index().assign(**{self.metric_name:np.nan})
                # else:
                #     self.data = self.dm.test_dataset.data.reset_index().assign(**{f"{self.metric_name}_{i}":np.nan for i in range(metric_size)})

            current_idx = pl_module.hparams.batch_size*batch_idx+sample_idx
            print(current_idx)
            print(value_names)
            print(sample_pred)
            print(len(sample_pred))
            self.data.loc[current_idx, value_names] = sample_pred
            print(self.data.loc[current_idx, value_names])

            # if  metric_size == 1:
            #     self.data.loc[current_idx, self.metric_name] = sample_pred.item()
            # else:
            #     cols = [f"{self.metric_name}_{i}" for i in range(metric_size)]
            #     self.data.loc[current_idx, cols] = sample_pred.detach().cpu().flatten().numpy()

    def on_test_epoch_end(self, trainer, pl_module):
        if not self.per_sample:
            pred = pd.DataFrame(self.data[0])
            truth = pd.DataFrame(self.data[1])
            pred.to_csv(f"{self.score_file.split('_scores')[0]}-gpu{self.gpu}-reconstruction-pred.csv")
            truth.to_csv(f"{self.score_file.split('_scores')[0]}-gpu{self.gpu}--reconstruction-truth.csv")

            pred = torch.from_numpy(data["pred"].values).to(torch.int64)
            pred[pred<0] = self.num_classes
            pred = F.one_hot(pred, num_classes=self.num_classes).view(-1, self.num_classes+1)
            pred = pred[pred[:, -1]==0][:, :-1]

            truth = torch.from_numpy(data["truth"].values).to(torch.int64)
            truth[truth<0] = self.num_classes
            truth = F.one_hot(truth, num_classes=self.num_classes).view(-1, self.num_classes+1)
            truth = truth[truth[:, -1]==0][:, :-1]


            if len(self.data.dropna()) != len(self.data):
                print(self.data)
                print([c for c in self.data.columns if c.startswith("pred_") or c.startswith("pred_")])
                self.data = self.data[[c for c in self.data.columns if c.startswith("pred_") or c.startswith("true_")]]
            self.data.to_pickle(f"{self.score_file.split('_scores')[0]}-reconstruction.csv")
            self.full_data = self.data.copy()
            pred = torch.from_numpy(self.data.loc[:, [c for c in self.data.columns if c.startswith("pred_")]].values).int()
            truth = torch.from_numpy(self.data.loc[:, [c for c in self.data.columns if c.startswith("true_")]].values).int()

            if self.num_classes > 1:
                pred = pred.view(-1, self.num_classes)
                true = true.view(-1, self.num_classes)

            m = self.metric_func(num_classes=pred.shape[1])
            m.update(pred, truth)
            result = m.compute()
            if self.metric == "ROC":
                fpr, tpr, thresholds = result
                auroc = auc(fpr, tpr)
                self.data = pd.DataFrame({
                    **{f"tpr_{i}":v for i, v in enumerate(tpr)},
                    **{f"fpr_{i}":v for i, v in enumerate(fpr)},
                    **{f"thresholds_{i}":v for i, v in enumerate(tpthresholdsr)},
                    "auroc": auroc
                })

        self.data.to_csv(self.score_file)

    def roc_from_csv(self, prefix, num_classes=21):
        prediction = None
        labels = None
        for f in glob.glob(f"{prefix}*-reconstruction-pred.csv"):
            pred = pd.read_csv(f)
            pred["0"] = pred["0"].apply(eval)
            pred = pred["0"].apply(pd.Series)
            pred = torch.from_numpy(pred.values).to(torch.int64)
            pred[pred<0] = 20
            pred = F.one_hot(pred, num_classes=num_classes).view(-1, 21)[:, :-1]
            pred = torch.cat((pred, torch.zeros((pred.size()[0],1))), 1)
            pred[pred.sum(dim=1)==0, -1] = 1
            if prediction is None:
                prediction = pred
            else:
                torch.cat((prediction, pred), axis=0)

        for f in glob.glob(f"{prefix}*-reconstruction-truth.csv"):
            truth = pd.read_csv(f).drop(columns=["Unnamed: 0"])
            truth = torch.from_numpy(truth.values).to(torch.int64)
            truth[truth<0] = 20
            truth = F.one_hot(truth, num_classes=num_classes).view(-1, 21)[:, :-1]
            truth = torch.cat((truth, torch.zeros((truth.size()[0],1))), 1)
            truth[truth.sum(dim=1)==0, -1] = 1
            _, truth = torch.max(truth, axis=1)
            if labels is None:
                labels = truth
            else:
                torch.cat((labels, truth), axis=0)

        roc = ROC(num_classes=1)(prediction.ravel(), labels.ravel())
        auroc = AUROC(num_classes=1)(prediction.ravel(), labels.ravel())

        if self.num_classes>1:
            multi_fpr, mult_tpr, multi_threshold = ROC(num_classes=21)(prediction, labels)
            maulti_auroc = AUROC(num_classes=21)(prediction, labels)

            # First aggregate all false positive rates
            all_fpr = torch.cat([multi_fpr[i] for i in range(n_classes)]).unique()

            # Then interpolate all ROC curves at this points
            mean_tpr = torch.zeros_(all_fpr)
            for i in range(num_classes):
                mean_tpr += interp(all_fpr, multi_fpr[i], mult_tpr[i])

            mean_tpr /= num_classes


        return pred, truth, roc, auroc

    def get_numpy(self):
        return self.data[[c for c in self.data.columns if self.metric_name in c]].values
