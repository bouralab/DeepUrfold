import os
import copy
import argparse
from functools import partial

from torchmetrics import classification as classification_metrics
from torchmetrics.functional.classification import auc

from DeepUrfold.Metrics.MetricSaver import MetricSaver, custom_metrics, cat_ravel, \
    one_hot_cat_ravel, clamp_cat_ravel, cat, one_hot_cat

import torch
import MinkowskiEngine as ME

from sklearn import metrics as skmetrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pytorch_lightning.utilities import rank_zero_only

class ClassifactionMetricSaver(MetricSaver):
    def __init__(self, hparams, prefix=None, num_classes=None, ax=None, multiple_figs=False, per_sample=False):
        super().__init__(hparams, prefix=prefix)
        if ax is None:
            ax = {}
        self.ax = ax
        self.multiple_figs = multiple_figs
        self.num_classes = num_classes if num_classes is not None else len(hparams.features)

        self.metrics["roc"] = classification_metrics.ROC(num_classes=1, pos_label=1, compute_on_step=False)
        #self.metrics["auroc"] = custom_metrics.AUROC(num_classes=1, compute_on_step=False)
        self.metrics["f1"] = classification_metrics.F1(num_classes=1, compute_on_step=False)
        self.metrics["prc"] = classification_metrics.PrecisionRecallCurve(num_classes=1, compute_on_step=False)
        self.metrics["auprc"] = classification_metrics.AveragePrecision(num_classes=1, compute_on_step=False)

        if self.num_classes > 1:
            self.metrics["multi_roc"] = classification_metrics.ROC(num_classes=self.num_classes, pos_label=1, compute_on_step=False)
            #self.metrics["multi_auroc"] = custom_metrics.AUROC(num_classes=self.num_classes, compute_on_step=False)
            #self.metrics["multi_f1"] = classification_metrics.F1(num_classes=self.num_classes, compute_on_step=False)
            #self.metrics["multi_prc"] = classification_metrics.PrecisionRecallCurve(num_classes=self.num_classes, compute_on_step=False)
            #self.metrics["multi_auprc"] = classification_metrics.AveragePrecision(num_classes=self.num_classes, compute_on_step=False)

        if self.num_classes > 1:
            self.use_other_metric["roc"] = {
                "multi_roc":[
                    ("preds", cat_ravel),
                    ("target", partial(one_hot_cat, ravel=not self.per_sample, num_classes=self.num_classes))
                ]
            }
            # self.use_other_metric["auroc"] = {
            #     "multi_roc":[
            #         ("preds", cat_ravel),
            #         ("target", partial(one_hot_cat_ravel, num_classes=self.num_classes))
            #     ]
            # }
            self.use_other_metric["f1"] = {
                "multi_roc":[
                    ("preds", clamp_cat_ravel),
                    ("target", partial(one_hot_cat, ravel=not self.per_sample, num_classes=self.num_classes))
                ]
            }
            self.use_other_metric["prc"] = {
                "multi_roc":[
                    ("preds", cat_ravel),
                    ("target", partial(one_hot_cat, ravel=not self.per_sample, num_classes=self.num_classes))
                ]
            }
            self.use_other_metric["auprc"] = {
                "multi_roc":[
                    ("preds", cat_ravel),
                    ("target", partial(one_hot_cat, ravel=not self.per_sample, num_classes=self.num_classes))
                ]
            }
            # self.use_other_metric["multi_auroc"] = {
            #     "multi_roc":[
            #         ("preds", cat_ravel),
            #         ("target", cat_ravel)
            #     ]
            # }
            # self.use_other_metric["multi_f1"] = {
            #     "multi_roc":[
            #         ("preds", cat_ravel),
            #         ("target", partial(one_hot_cat, num_classes=self.num_classes))
            #     ]
            # }
            # self.use_other_metric["multi_prc"] = {
            #     "multi_roc":[
            #         ("preds", cat_ravel),
            #         ("target", cat_ravel)
            #     ]
            # }
            # self.use_other_metric["multi_auprc"] = {
            #     "multi_prc":[
            #         ("preds", cat_ravel),
            #         ("target", partial(one_hot_cat, ravel=not self.per_sample,))
            #     ]
            # }
            self.save_metric["multi_roc"] = ["preds", "target"]
        else:
            #self.use_other_metric["auroc"] = {"roc":["preds", "target"]}
            self.use_other_metric["f1"] ={
                "roc":[
                    ("preds", lambda x: torch.clamp(cat(x), min=0, max=1)),
                    ("target", cat)
                ]
            }
            self.use_other_metric["prc"] = {
                "roc":[
                    ("preds", cat),
                    ("target", cat)
                ]
            }
            self.use_other_metric["auprc"] = {
                "roc":[
                    ("preds", cat),
                    ("target", cat)
                ]
            }
            self.save_metric["roc"] = ["preds", "target"]

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
            labels = labels[:, :data.size()[1]]
            del data

        truth = ME.SparseTensor(labels.float(), coords.int(), device=device)
        labels = truth.F

        if labels.size()[1] > 1:
            if self.group_feats_start is None:
                _, labels = torch.max(labels, dim=1)
            else:
                _labels = None
                print(self.group_feats_start)
                for i in range(len(self.group_feats_start)-1):
                    start, end = self.group_feats_start[i], self.group_feats_start[i+1]
                    l = labels[:, start:end]
                    if l.size()[1] > 1:
                        l = torch.max(l, dim=1)[1].unsqueeze(0).T

                    if _labels is not None:
                        _labels = torch.cat((_labels, l), dim=1)
                    else:
                        _labels = l

                labels = _labels

        i = 0
        for metric_name, metric_func in self.metrics.items():
            if metric_name in self.use_other_metric: continue

            if multiple_results:
                r = result[i]
            else:
                r = result

            if self.per_sample:
                for sample_pred, sample_truth in zip(r.decomposed_features, truth.decomposed_features):
                    metric_func(sample_pred, sample_truth)
            else:
                if self.group_feats_start is None:
                    metric_func(r.F, labels) #pred
                elif metric_name in self.save_metric and self.save_metric[metric_name] == ["preds", "target"]:
                    metric_func.preds.append(r.F)
                    metric_func.target.append(labels)
                else:
                    raise RuntimeError("Error in groups")


            i += 1

    @rank_zero_only
    def process(self, trainer=None, pl_module=None, stage=None, save=True, no_compute=False):
        super().process(trainer=trainer, pl_module=pl_module, stage=stage, save=save, no_compute=no_compute)
        if not self.per_sample and ("inside_feature_group" not in self.hparams or not self.hparams.inside_feature_group):
            # feature_groups = self.hparams.feature_groups
            # self.hparams.feature_groups = None
            # self.from_preds_target(
            #     "classification-multi_roc-preds.pt",
            #     "classification-multi_roc-target.pt",
            #     self.hparams,
            #     prefix=self.prefix,
            #     num_classes=1)
            # self.hparams.feature_groups = feature_groups
            features = self.hparams.features
            if hasattr(self.hparams, "features_original"):
                self.hparams.features = [self.hparams.features_original]

            assert self.hparams.feature_groups is not None
            print("FEATURE GROUPS", self.hparams.feature_groups, ("inside_feature_group" not in self.hparams or not self.hparams.inside_feature_group))
            ClassifactionMetricSaver.from_separated_preds_target(
                "classification-multi_roc-preds.pt",
                "classification-multi_roc-target.pt",
                self.hparams,
                prefix=self.prefix,
                num_classes=len(features))
            if hasattr(self.hparams, "features_original"):
                features = self.hparams.features
                self.hparams.features = features

    # def process_epoch(self, **metrics):
    #     self.compute_values()

    @classmethod
    def from_separated_preds_target(cls, preds, target, hparams, prefix=None, num_classes=None, ax=None, multiple_figs=False):
        if isinstance(preds, str) and os.path.isfile(preds):
            preds = torch.load(preds) #.to("cuda")
        if isinstance(target, str) and os.path.isfile(target):
            target = torch.load(target) #.to("cuda")

        assert isinstance(preds, torch.Tensor) and isinstance(target, torch.Tensor)

        if not isinstance(preds, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise RuntimeError("preds and target must be a tensors or files for torch.load")

        if hparams.feature_groups is not None and ("inside_feature_group" not in hparams or not hparams.inside_feature_group):
            print("HERE")
            if len(hparams.feature_groups) == 1:
                hparams.feature_groups = hparams.feature_groups[0].split(";")

            if hparams.features is not None:
                if len(hparams.features) == 1:
                    hparams.features = hparams.features[0]
                else:
                    raise RuntimeError("Must use ';' to seprate feature names and '__' to seperate feature groups")
            else:
                hparams.features = [f"feat{i}" for i in range(num_features)]



            group_feats = [f.split(";") for f in hparams.features.split("__")]
            assert len(group_feats)==len(hparams.feature_groups)
            assert sum(len(f) for f in group_feats)==num_classes, f"{sum(len(f) for f in group_feats)} == {num_classes}"
            feat_start = 0
            target2 = target
            target = target.view(preds.size()[0], len(hparams.feature_groups))
            for group_num, (group_name, group_features) in enumerate(zip(hparams.feature_groups, group_feats)):
                print(group_name, group_features, preds.size(), target.size())
                p = preds[:, feat_start:feat_start+len(group_features)]
                t = target[:, group_num].int()

                if len(group_features) == 1:
                    p = p.squeeze(1)

                print("Running with preds", p.size(), "targets", t.size())

                hparams.prefix = f"{prefix} {group_name}"
                hparams.features = group_features
                hparams.inside_feature_group = True
                try:
                    hp = copy.deepcopy(hparams)
                    hp.feature_groups = None

                    cls.from_preds_target(p, t, hp, prefix=hparams.prefix,
                        num_classes=len(group_features), ax=ax, multiple_figs=multiple_figs)

                    if len(group_features) > 1:
                        hparams.prefix = f"{prefix} {group_name} Separated"

                        cls.from_separated_preds_target(p, t, hp, prefix=hparams.prefix,
                            num_classes=len(group_features), ax=ax, multiple_figs=multiple_figs)

                except ValueError:
                    raise
                    print("Failed for feature", group_name)
                    import pdb; pdb.set_trace()
                feat_start += len(group_features)


            expanded_target = torch.Tensor()
            feat_start = 0
            for group_num, (group_name, group_features) in enumerate(zip(hparams.feature_groups, group_feats)):
                print(group_name, group_features, preds.size(), target.size())
                p = preds[:, feat_start:feat_start+len(group_features)]
                t = target[:, group_num].int()

                if len(group_features) == 1:
                    new_target = t.unsqueeze(1)
                else:
                    new_target = one_hot_cat(t, num_classes=len(group_features))

                print("ADDING", new_target, new_target.size())
                expanded_target = torch.cat((expanded_target, new_target), axis=1)

                feat_start += len(group_features)

            print(expanded_target, expanded_target.size(), expanded_target.max())
            print(preds, preds.size(), preds.max())

            hparams.prefix = f"{prefix} "
            hparams.inside_feature_group = True
            try:
                hp = copy.deepcopy(hparams)
                hp.feature_groups = None
                cls.from_preds_target(preds, expanded_target, hp, prefix=f"{hparams.prefix} All Combined",
                    num_classes=1, ax=ax, multiple_figs=multiple_figs)

            except ValueError:
                raise
                print("Failed for feature", group_name)
                import pdb; pdb.set_trace()

            return


        if hparams.features is not None:
            if len(hparams.features) == 1:
                hparams.features = hparams.features[0].split(";")

            assert len(hparams.features)==num_classes, f"{len(hparams.features)} == {num_classes}"
            features = hparams.features
        else:
            features = [f"feat{i}" for i in range(num_features)]

        target = one_hot_cat(target, num_classes=num_classes)

        ax = None
        bar_df = None
        for i, feature in enumerate(features):
            print(feature)
            p = preds[:,i].view(-1)
            t = target[:,i].view(-1)
            print(p.size(), t.size())
            print(p)
            print(t)
            print("SUM", t.sum())
            if t.sum()==0:
                print("skipped", feature)
                continue
                p = 1-p
                t = 1-t
            classification_metrics = cls(hparams, prefix=feature, num_classes=1, ax=ax, multiple_figs=True)
            classification_metrics.metrics["roc"](p, t)
            classification_metrics.process(save=False) #on_test_epoch_end(None, None, save=False)
            classification_metrics.compute_values()
            if ax is None:
                ax = classification_metrics.ax

            if "bar" in classification_metrics.metric_values:
                if bar_df is None:
                    bar_df = classification_metrics.metric_values["bar"]
                else:
                    bar_df = pd.concat((bar_df, classification_metrics.metric_values["bar"]), axis=0)


        if "bar" in classification_metrics.metric_values:
            bar_df.to_csv(f"{prefix}-metrics.csv")

        for metric, (fig, ax) in ax.items():
            fig.savefig(f"{prefix}-{metric}.pdf")

    @classmethod
    def from_preds_target(cls, preds, target, hparams, prefix=None, num_classes=None, ax=None, multiple_figs=False):

        if isinstance(preds, str) and os.path.isfile(preds):
            preds = torch.load(preds)#.to("cuda")
        if isinstance(target, str) and os.path.isfile(target):
            target = torch.load(target)#.to("cuda")

        if not isinstance(preds, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise RuntimeError("preds and target must be a tensors or files for torch.load")


        classification_metrics = cls(hparams, prefix=prefix, num_classes=num_classes, ax=ax, multiple_figs=multiple_figs)

        if num_classes > 1:
            classification_metrics.metrics["multi_roc"](preds, target.int())
        else:
            classification_metrics.metrics["roc"](preds, target.int())

        #Copy pred, target to all metrics
        classification_metrics.process(save=False)
        #classification_metrics.on_test_epoch_end(None, None, save=False)
        classification_metrics.compute_values()

        torch.cuda.empty_cache()

        return classification_metrics

    @classmethod
    def from_multiple_runs(cls, info_file, hparams, prefix=None):
        runs = pd.read_csv(info_file)
        ax = None
        bar_df = None
        for i, run in runs.iterrows():
            print("RUN", run["name"])
            print(run)
            print(os.path.join(run.dir, run.preds))
            params = argparse.Namespace(**vars(hparams))
            print(run.extra_args)
            if isinstance(run.extra_args, str):
                for extra_arg in run.extra_args.split(";"):
                    key, value = extra_arg.split("=")
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    setattr(params, key, value)
            print(params)

            classification_metrics = cls.from_preds_target(os.path.join(run.dir, run.preds), os.path.join(run.dir, run.target), params, prefix=run["name"], num_classes=run.num_classes, ax=ax, multiple_figs=True)
            if ax is None:
                ax = classification_metrics.ax

            if "bar" in classification_metrics.metric_values:
                if bar_df is None:
                    bar_df = classification_metrics.metric_values["bar"]
                else:
                    bar_df = pd.concat((bar_df, classification_metrics.metric_values["bar"]), axis=0)

        if "bar" in classification_metrics.metric_values and bar_df is not None:
            bar_fig, bar_ax = plt.subplots(figsize=(8,8))
            g = sns.barplot(x="Feature", y="Value", hue="Metric", data=bar_df, ax=bar_ax)
            plt.xticks(rotation=70, figure=bar_fig)
            g.text(bar_df.name,row.tip, round(row.total_bill,2), color='black', ha="center")

            handles, labels = bar_ax.get_legend_handles_labels()
            bar_ax.legend(handles, labels, title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
            ax["bar"] = (bar_fig, bar_ax)
            bar_df.to_csv(f"{prefix}-all-metrics.csv")

        for metric, (fig, ax) in ax.items():
            if metric in ["roc", "prc"]:
                handles, labels = ax.get_legend_handles_labels()
                # sort both labels and handles by labels
                labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0].split(" = ")[-1][:-1]), reverse=True))
                ax.legend(handles, labels)
                ax.legend(handles, labels, title='Features', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')

            plt.tight_layout(rect=[0,0,0.75,1])

            fig.savefig(f"{prefix}-{metric}.pdf", bbox_inches="tight")

    def compute_values(self, metrics=None, ax=None, plotting=True):
        print("plot")
        if metrics is None:
            metrics = list(self.metrics.keys())

        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]

        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()

        # params = {
        #    'axes.labelsize': 8,
        #    'text.fontsize': 8,
        #    'legend.fontsize': 10,
        #    'xtick.labelsize': 10,
        #    'ytick.labelsize': 10,
        #    'text.usetex': False,
        #    'figure.figsize': [8,8]
        #    }
        # rcParams.update(params)

        print(metrics)

        if "roc" in metrics:
            fpr, tpr, thresholds = self.metric_values["roc"]
            roc_auc = [v for n, v in self.metric_values.items() if "auroc" in n and not n.startswith("multi")]
            roc_auc = roc_auc[0] if len(roc_auc) > 0 else None
            print(self.metrics["roc"].preds)
            roc_auc = skmetrics.roc_auc_score(
                cat_ravel(self.metrics["roc"].target).cpu().int().numpy(),
                cat(self.metrics["roc"].preds).cpu().numpy(),
                )
            self.metric_values["auroc"] = roc_auc

            if self.per_sample:
                roc_auc = [skmetrics.roc_auc_score(
                    cat_ravel(self.metrics["roc"].target[i]).cpu().int().numpy(),
                    cat(self.metrics["roc"].preds[i]).cpu().numpy(),
                    ) for i in range(self.metrics["roc"].target.shape())]

            if self.multiple_figs:
                estimator_name = self.prefix
            else:
                estimator_name = "micro-average ROC curve" if self.num_classes > 1 else None

            fig, ax = self.ax.get("roc", plt.subplots(figsize=(8,8)))
            roc = skmetrics.RocCurveDisplay(fpr=fpr.cpu().numpy(), tpr=tpr.cpu().numpy(), #
                roc_auc=roc_auc, estimator_name=estimator_name)
            roc.plot(ax=ax)
            self.ax["roc"] = (fig, ax)

            print("MFIG", self.multiple_figs)

            if not self.multiple_figs:
                if self.num_classes > 1:
                    fig.savefig(f"{self.prefix}-{estimator_name}.pdf")
                    print("saved fig to", os.getcwd())
                else:
                    fig.savefig(f"{self.prefix}-ROC_curve.pdf")

        if "prc" in metrics:
            precision, recall, thresholds = self.metric_values["prc"]
            average_precision = [v for n, v in self.metric_values.items() if "auprc" in n and not n.startswith("multi")]
            average_precision = average_precision[0].cpu().numpy() if len(average_precision) > 0 else None

            if self.multiple_figs:
                estimator_name = self.prefix
            else:
                estimator_name = "micro-average PRC curve" if self.num_classes > 1 else None

            fig, ax = self.ax.get("prc", plt.subplots(figsize=(8,8)))
            prc = skmetrics.PrecisionRecallDisplay(precision=precision.cpu().numpy(), recall=recall.cpu().numpy(),
                estimator_name=estimator_name, average_precision=average_precision)
            prc.plot(ax=ax)
            self.ax["prc"] = (fig, ax)

            if not self.multiple_figs:
                if self.num_classes > 1:
                    fig.savefig(f"{self.hparams.prefix}-{estimator_name}.pdf")
                else:
                    fig.savefig(f"{self.hparams.prefix}-PRC_curve.pdf")

        if "auroc" in metrics or "auprc" in metrics or "f1" in metrics:
            metrics = pd.DataFrame({
                "Feature":[self.prefix]*3,
                "Metric":["AUROC", "AUPRC", "F1"],
                "Value":[
                    self.metric_values.get("auroc", np.nan),
                    self.metric_values["auprc"].cpu().item() if "auprc" in self.metric_values else np.nan,
                    self.metric_values["f1"].cpu().item() if "f1" in self.metric_values else np.nan
                    ]
                })
            print("METRICS", metrics)
            print("METRICS, mfig", self.multiple_figs)
            self.metric_values["bar"] = metrics


            if not self.multiple_figs:
                print("Saving metrics as bar and csv")
                fig, ax = plt.subplots(figsize=(8,8))
                sns.barplot(x="Feature", y="Value", hue="Metric", data=metrics, ax=ax)
                fig.savefig(f"{self.hparams.prefix}-bar-plot.pdf")
                metrics.to_csv(f"{self.hparams.prefix}-metrics.csv")
                print("Saving metrics as bar and csv", f"{self.hparams.prefix}-metrics.csv")


        if False and "multi_roc" in metrics:
            multi_fpr, _ = self.metric_values["multi_roc"]
            roc_auc2 = [v for n, v in self.metrics.items() if "auroc" in n and n.startswith("multi")]
            roc_auc2 = roc_auc[0] if len(auc) > 0 else None

            # First aggregate all false positive rates
            all_fpr = torch.cat([multi_fpr[i] for i in range(self.num_classes)]).unique()

            # Then interpolate all ROC curves at this points
            mean_tpr = torch.zeros_(all_fpr).numpy()
            for i in range(num_classes):
                mean_tpr += interp(all_fpr, multi_fpr[i], mult_tpr[i])

            mean_tpr /= num_classes

            fpr, tpr = all_fpr, mean_fpr
            roc_auc = auc(torch.from_numpy(fpr), torch.from_numpy(tpr))
            roc_auc = (roc_auc, roc_auc2)

            estimator_name = "macro-avarge ROC curve" if self.num_classes > 1 else None

            fig, ax = plt.subplots()
            metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                estimator_name=estimator_name, ax=ax)
            fig.savefig(f"{self.hparams.prefix}-ROC_curve.pdf")

        if "multi_prc" in metrics:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metric", nargs="+", default=None)
    parser.add_argument("-p", "--prefix", default="ClassifactionMetricSaver")

    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--preds", default=None)
    inputs.add_argument("--runs", default=None)

    parser.add_argument("--target", required=False, default=None)
    parser.add_argument("-n", "--num_classes", default=0, type=int)

    parser.add_argument("--sep", default=False, action="store_true")
    parser.add_argument("--features", default=None, nargs="+")
    parser.add_argument("--feature_groups", default=None, nargs="+")
    parser.add_argument("--gt0", default=False, action="store_true")

    args = parser.parse_args()

    if args.preds is not None:
        if args.target is None or args.num_classes == 0:
            raise RuntimeError("If preds are used as input, you must also specify target and num_classes")
        if args.sep:
            ClassifactionMetricSaver.from_separated_preds_target(args.preds,
                args.target, args, prefix=args.prefix, num_classes=args.num_classes)
        else:
            ClassifactionMetricSaver.from_preds_target(args.preds, args.target,
                args, prefix=args.prefix, num_classes=args.num_classes)
    elif args.runs is not None:
        if args.target is not None or args.num_classes != 0:
            print("Ignoring target and num_classes becuase runs was sepecied")
        ClassifactionMetricSaver.from_multiple_runs(args.runs, args, prefix=args.prefix)
