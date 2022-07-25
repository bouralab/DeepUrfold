import os
import argparse
from collections import OrderedDict, defaultdict

import torch
import wandb
import numpy as np
import pytorch_lightning as pl
import MinkowskiEngine as ME

import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset

from DeepUrfold.Datasets.DomainStructureDataset import DomainStructureDataset
from DeepUrfold.util import str2bool
#from scorer import ProteinScorer

#from MinkowskiEngine.modules import minkunet
from DeepUrfold.Models import minkunet
unet_models = {name:getattr(minkunet, name) for name in dir(minkunet) \
    if name.startswith("MinkUNet") and "Base" not in name}

def init_weights(m):
    if isinstance(m, (scn.BatchNormLeakyReLU, scn.BatchNormReLU)):
        return
    if hasattr(m, "weight"):
        torch.nn.init.kaiming_uniform_(m.weight)

def mem_report(*tensors):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        LEN = 65
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('%s\t\t%s\t\t%.2f' % (
                element_type,
                size,
                mem) )
        print('-'*LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
        print('-'*LEN)


class DomainStructureSegmentationModel(pl.LightningModule):
    LATENT_VARS = ["latent"]

    def __init__(self, hparams, metrics=None):
        # init superclass
        super().__init__()

        self.hparams = hparams

        if "tpu_cores" in self.hparams:
            del self.hparams["tpu_cores"]

        if isinstance(self.hparams.features_to_drop, (list, tuple)) and \
          len(self.hparams.features_to_drop) > 0:
            self.hparams.features = [feature for feature in self.hparams.features if \
                feature not in self.hparams.features_to_drop]

        if self.hparams.nClasses is None or self.hparams.autoencoder:
            self.hparams.nClasses = len(self.hparams.features)
        else:
            try:
                self.hparams.nClasses = int(self.hparams.nClasses)
            except ValueError:
                if self.hparams.nClasses in ["bool", "bool1"]:
                    self.hparams.nClasses = 1
                elif self.hparams.nClasses == "bool2":
                    self.hparams.nClasses = 2
                elif self.hparams.label_type == "sfam":
                    self.hparams.nClasses = 6119
                else:
                    self.hparams.nClasses = 1

        unet = unet_models.get(self.hparams.unet, "MinkUNet34B")

        self.nn_model = unet(len(self.hparams.features), self.hparams.nClasses,
                             leakiness = self.hparams.leakiness,
                             dropout_p = self.hparams.dropout,
                             skipconnections = self.hparams.skip_connections,
                             D=3)

        if self.hparams.distributed_backend is not None:
            self.nn_model = self.configure_sync_batchnorm(self.nn_model)

        self.hparams.report_metrics_n_batches = max(1, self.hparams.report_metrics_n_batches)

        #self.nn_model.apply(init_weights)

        #self.current_epoch = -1
        #self.media_dir = ""

        self.metrics = metrics if metrics is not None else {}
            #(metric, use_bool)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, inputs):
        return self.nn_model(inputs).F.contiguous().double() #.to(self.device)

    # def on_train_start(self):
    #     self.current_epoch += 1
    #     self.media_dir = os.path.join("media", "epoch{}".format(self.current_epoch))
    #     self.metrics = {name: (metric.to(self.device), use_bool) for name, (metric, use_bool) in self.metrics.items()}
    #     return super().on_train_start()

    def loss(self, outputs, labels, bfactors=None):
        #From Chung, Wang, Bourne. "Exploiting sequence and structure homologs to identify
        #protein–protein binding sites." Proteins: Structure, Function, Bioinformatics. 2005.
        #https://doi.org/10.1002/prot.20741
        loss = F.mse_loss(outputs, labels) #labels.F.contiguous().double()) #labels.to(self.device).double())
#         if self.hparams.normalize_by_b:
#             b = bfactors.unsqueeze(1).repeat(1, outputs.size()[1]).to(outputs.device).exp().pow(-1)
#             b = ME.SparseTensor(
#                 # coords=coords,  not required
#                 feats=b,
#                 coords_manager=outputs.coords_man,  # must share the same coordinate manager
#                 coords_key=outputs.coords_key  # For inplace, must share the same coords key
#             )

#             outputs = b*outputs
#             del b

        return loss

    def __sync_across_gpus(self, t):   # t is a tensor

        gather_t_tensor = [torch.ones_like(t) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(gather_t_tensor, t)
        return torch.cat(gather_t_tensor)

    def __compute_metrics(self, outputs, labels):
        _outputs = torch.max(outputs, 1)[1].to(self.device) #(outputs>=0.7).int()
        _labels = torch.max(labels.int(), 1)[1].to(self.device) #np.argmax(labels.astype(int), axis=1)

        # sync across gpus
        if False and self.trainer.use_ddp:
            _outputs = self.__sync_across_gpus(_outputs)
            _labels = self.__sync_across_gpus(_labels)

        metrics = {}
        for name, (metric, use_bool) in self.metrics.items():
            #metric = metric.to(self.device)
            value = metric(_outputs if use_bool else outputs, _labels)
            metrics[name] = value
        return metrics

        metrics = {name: metric.to(self.device)(_outputs if use_bool else outputs, _labels) \
            for name, (metric, use_bool) in self.metrics.items()}

        if False and self.trainer.use_ddp:
            metrics = {name:value/self.trainer.world_size for name, value in \
                metrics.items()}

        return metrics

    def training_step(self, batch, batch_num):
        coords, feats, labels = batch

        # np.save("coords{}.pt".format(batch_num), coords.cpu().detach().numpy())
        # np.save("feats{}.pt".format(batch_num), feats.cpu().detach().numpy())
        # np.save("labels{}.pt".format(batch_num), labels.cpu().detach().numpy())
        #
        # return torch.rand(1, 1, requires_grad=True)

        if ME.__version__ == '0.4.3' and coords.is_cuda:
            coords = coords.cpu()

        feats = ME.SparseTensor(feats.float(), coords.int())
        outputs = self.forward(feats)

        #labels = labels.to(outputs.device).double()

        loss = self.loss(outputs, labels)

        if batch_num % self.hparams.report_metrics_n_batches == 0 and False:
            metrics = self.__compute_metrics(outputs, labels)
        else:
            metrics = {}

        # del feats, coords, labels
        # del outputs
        # del batch

        return loss

        # result = pl.TrainResult(minimize=loss)
        # result.log('train_loss', loss, logger=True, prog_bar=True, on_step=True, sync_dist=False)
        # if metrics:
        #     result.log_dict(metrics, logger=True, prog_bar=True, on_step=True, sync_dist=False)
        #
        # return result


    # def training_end(self, batch):
    #     print("TRAIN BATCH IS {}".format(batch))
    #     batch = {name: value.mean().to(self.device) for name, value in batch.items()}
    #     loss = batch["loss"]
    #     del batch["loss"]
    #
    #     result = pl.TrainResult(minimize=loss)
    #     result.log('train_loss', loss, prog_bar=True, on_step=True)
    #
    #     for metric, value in batch.items():
    #         result.log(metric, value, prog_bar=True, on_step=True)
    #
    #     del batch
    #     return result

    def validation_step(self, batch, batch_num):
        coords, feats, labels = batch
        if coords.is_cuda:
            coords = coords.cpu()

        feats = ME.SparseTensor(feats.float(), coords.int())
        outputs = self.forward(feats)

        loss = self.loss(outputs, labels) #truth, b_factors)
        if batch_num % self.hparams.report_metrics_n_batches == 0:
            metrics = self.__compute_metrics(outputs, labels)
        else:
            metrics = {}

        # del feats
        # del outputs

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, logger=True, prog_bar=True, on_step=True, sync_dist=True)
        if metrics:
            result.log_dict(metrics, logger=True, prog_bar=True, on_step=True, sync_dist=True)

        return result

    # def validation_step_end(self, batch):
    #     batch = {name: value.mean() for name, value in batch.items()}
    #     loss = batch["loss"]
    #     del batch["loss"]
    #
    #     result = pl.EvalResult(checkpoint_on=loss)
    #     result.log('val_loss', loss, prog_bar=True)
    #
    #     for metric, value in batch.items():
    #         result.log(metric, value, prog_bar=True)
    #
    #     del batch
    #     return result

    # def validation_epoch_end(self, outputs):
    #     loss_mean = sum(output['val_loss'] for output in outputs)
    #     loss_mean /= len(outputs)
    #
    #     tqdm_dict = {'val_loss': loss_mean.item()}
    #
    #     # show val_acc in progress bar but only log val_loss
    #     results = {
    #         'progress_bar': tqdm_dict,
    #         'log': {'val_loss': loss_mean}
    #     }
    #     return results

    def test_step(self, batch, batch_num):
        coords, feats, labels = batch
        if coords.is_cuda:
            coords = coords.cpu()

        feats = ME.SparseTensor(feats.float(), coords.int())
        outputs = self.forward(feats)

        loss = self.loss(outputs, labels) #truth, b_factors)
        metrics = self.__compute_metrics(outputs, labels)

        # del feats, coords, labels
        # del outputs
        # del batch

        result.log('test_loss', loss, logger=True, prog_bar=True, on_step=True, sync_dist=True)
        if metrics:
            result.log_dict(metrics, logger=True, prog_bar=True, on_step=True, sync_dist=True)

        return result

    # def test_step_end(self, batch):
    #     batch = {name: value.mean() for name, value in batch.items()}
    #     loss = batch["loss"]
    #     del batch["loss"]
    #
    #     result = pl.EvalResult(checkpoint_on=loss)
    #     result.log('test_loss', loss, prog_bar=True)
    #
    #     for metric, value in batch.items():
    #         result.log(metric, value, prog_bar=True)
    #
    #     del batch
    #     return result

    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.cuda.empty_cache()

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        torch.cuda.empty_cache()

    def on_test_batch_end(self, outputs, batch, batch_idx):
        torch.cuda.empty_cache()

    def configure_sync_batchnorm(self, model):
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        return model

    def configure_ddp(self, model, device_ids):
        from DeepUrfold.Models.DataParallel import MinkowskiDistributedDataParallel
        model = MinkowskiDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=True
        )
        return model

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(),
            lr           = self.hparams.learning_rate,
            momentum     = self.hparams.momentum,
            weight_decay = self.hparams.weight_decay,
            nesterov     = True
            )
        return [optimizer]

    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i,
    #   second_order_closure=None):

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.hparams.learning_rate * np.exp((1 - current_epoch) * \
                                                                    self.hparams.lr_decay)

        # update params
        optimizer.step()
        optimizer.zero_grad()

    # def configure_ddp(self, model, device_ids):
    #     from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
    #     #model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    #     model = LightningDistributedDataParallel(
    #         model,
    #         device_ids=device_ids,
    #         find_unused_parameters=True
    #     )
    #     return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser],
            add_help=False)

        parser.add_argument("--prefix", default="")

        model = parser.add_argument_group('model')
        model.add_argument("--unet", default="MinkUNet34B", choices=unet_models)
        model.add_argument("--dropout", type=float, default=0.5)
        model.add_argument("--skip_connections", type=str2bool, nargs='?',
                           const=True, default=False)
        model.add_argument("--leakiness", type=float, default=0)
        model.add_argument("--nLabels", type=int, default=None,
            help="Default is None, where all nLables will be the same as the input features for an autoencoder")
        model.add_argument("--normalize_by_b", type=str2bool, nargs='?',
                           const=True, default=False)
        model.add_argument("--sweep", type=str2bool, nargs='?',
                           const=True, default=False,
                           help="If running a sweep, tells data loader to use 2 fewer cores")
        model.add_argument("--report_metrics_n_batches", type=int, default=10)

        optim = parser.add_argument_group('optim')
        optim.add_argument("--learning_rate", type=float, default=1e-4,
                           help="learning rate, default is 1e-4")
        optim.add_argument("--momentum", type=float, default=0.9,
                           help="momentum, default is 0.9")
        optim.add_argument("--weight_decay", type=float, default=1e-4,
                            help="weigh decay, default is 1e-4")
        optim.add_argument("--lr_decay", type=float, default=4e-2)

        return parser
