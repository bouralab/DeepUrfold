import os
from argparse import ArgumentParser

# Must be imported before large libs
try:
    import open3d as o3d
except ImportError:
    #Unable to load Open3D, cannot visualize tensors during training
    pass
    #raise ImportError("Please install open3d with `pip install open3d`.")

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F

#Set number of threads for MinkowskiEngine
if not "OMP_NUM_THREADS" in os.environ:
    os.environ["OMP_NUM_THREADS"] = "16"

import MinkowskiEngine as ME

from DeepUrfold.Models.minkae import Encoder, Decoder
from DeepUrfold.util import str2bool

class DomainStructureAeKL(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    """

    def __init__(
        self,
        hparams
    ):

        super().__init__()

        self.save_hyperparameters(hparams)

        if "tpu_cores" in self.hparams:
            del self.hparams["tpu_cores"]

        #self.save_hyperparameters()

        try:
            self.hparams.kernel_size
        except AttributeError:
            self.hparams.kernel_size=3
            self.hparams.transpose_kernel_size=2
            self.hparams.stride=2

        try:
            self.hparams.return_reconstruction
        except AttributeError:
            self.hparams.return_reconstruction = False

        try:
            self.hparams.return_var
        except AttributeError:
            self.hparams.return_var = False

        try:
            self.hparams.return_latent
        except AttributeError:
            self.hparams.return_latent = False

        try:
            self.hparams.tensor_field
        except AttributeError:
            self.hparams.tensor_field = False

        if not self.hparams.expand_surface_data_loader:
            from DeepUrfold.Models.van_der_waals_surface import VanDerWallsSurface
            self.create_molecular_surface = VanDerWallsSurface(
                volume=self.hparams.input_size,
                features=self.hparams.features,
                algorithm=self.hparams.space_fill_algorithm,
                device=self.device
            )

        print("Using", len(self.hparams.features), "features:", self.hparams.features)

        self.encoder = Encoder(
            len(self.hparams.features),
            kernel_size=self.hparams.kernel_size,
            stride=self.hparams.stride)
        self.decoder = Decoder(len(self.hparams.features),
            kernel_size=self.hparams.kernel_size,
            transpose_kernel_size=self.hparams.transpose_kernel_size,
            stride=self.hparams.stride)

        if self.hparams.distributed_backend is not None:
            self.encoder = self.configure_sync_batchnorm(self.encoder)
            self.decoder = self.configure_sync_batchnorm(self.decoder)

    def forward(self, x, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encoder(x)
        dec = self.decode(quant) 
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def sample(self, means, log_vars):
        zs = means + torch.exp(0.5 * log_vars.F) * torch.randn_like(log_vars.F)
        return zs

    def generate(self, n_samples=1, mean=0, var=1):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(mean, std)
        z = p.sample([n_samples, 1024])
        return self.decoder(z)

    def loss(self, x_hat, x, means, log_vars):
        if x_hat.size()[1] < x.size()[1]:
            other_features = x[:, x_hat.size()[1]:]
            x = x[:, :x_hat.size()[1]]
        else:
            other_features = None

        recon_loss = F.mse_loss(x_hat, x)

        # log_qz = q.log_prob(z)
        # log_pz = p.log_prob(z)

        KLD = -0.5 * torch.mean(
            torch.mean(1 + log_vars - means.pow(2) - log_vars.exp(), 1))

        # kl = log_qz - log_pz
        # kl = kl.mean()

        loss = KLD + recon_loss

        return recon_loss, KLD, loss, other_features

    def samplewise_loss(self, x_hat, x, means, log_vars):
        try:
            recon_loss, KLD, loss, other_features = zip(*[self.loss(*x) for x in zip(
                    x_hat.decomposed_features,
                    x.decomposed_features,
                    means.decomposed_features,
                    log_vars.decomposed_features)])
            return torch.FloatTensor(recon_loss), torch.FloatTensor(KLD), torch.FloatTensor(loss), other_features
        except RuntimeError:
            #Fails for some nto sure why? in means
            coord_sort = torch.sort(means.C[:,0], 0)[1]
            means1 = means.F[coord_sort, :]
            assert coord_sort.numpy().tolist()==torch.sort(log_vars.C[:,0], 0)[1].numpy().tolist()
            logvars1 = log_vars.F[coord_sort, :]

            recon_loss, KLD, loss, other_features = zip(*[self.loss(*x) for x in zip(
                    x_hat.decomposed_features,
                    x.decomposed_features,
                    means1.unsqueeze(1),
                    logvars1.unsqueeze(1))])
            return torch.FloatTensor(recon_loss), torch.FloatTensor(KLD), torch.FloatTensor(loss), other_features


    def step(self, batch, batch_idx, eval=False):

        if self.hparams.expand_surface_data_loader:
            if eval:
                coords, feats, labels = batch
                labels = ME.SparseTensor(labels.float(), coords.int(), device=labels.device)
            else:
                coords, feats = batch[:2] #Make sure there is only two items, ignore the rest

            if ME.__version__ == '0.4.3' and coords.is_cuda:
                coords = coords.cpu()

            visualize_tensors = "visualize_tensors" in self.hparams and self.hparams.visualize_tensors
            tensors_to_vizualize = {}

            if self.hparams.tensor_field:
                tfield = ME.TensorField(
                    coordinates=coords.int(),
                    features=feats.float(),
                    device=feats.device,
                    )
                x = tfield.sparse()
                if visualize_tensors:
                    original = ME.SparseTensor(feats.float(), coords.int())
                    tensors_to_vizualize["Unused SparseTensor input"] = [t.clone().cpu() for t in original.decomposed_coordinates]
                    tensors_to_vizualize["TensorField input"] = [t.clone().cpu() for t in tfield.decomposed_coordinates]
                    tensors_to_vizualize["TensorField SparseTensor input"] = [t.clone().cpu() for t in x.decomposed_coordinates]
            else:
                x = ME.SparseTensor(feats.float(), coords.int())
                if visualize_tensors:
                    tensors_to_vizualize["SparseTensor input"] = [t.clone().cpu() for t in x.decomposed_coordinates]
        else:
            #Create molecular surface using KNN on GPU => Does not work yet
            coordinates_radii, feats, labels, lengths = batch
            x, out = self.create_molecular_surface(coordinates_radii, feats, lengths, device=self.device)
            coords, batch = None, None
            if self.hparams.compare_kdtree:
                knn_points = set([tuple(x) for x in x.C.cpu().tolist()])
                kdtree_dataloader_points = set([tuple(x) for x in labels[0].cpu().tolist()])

                if knn_points!=kdtree_dataloader_points:
                    import pdb; pdb.set_trace()
                    #newCoords = torch.sort(x.C, dim=0)[0].cpu()                                               │    Uninstalling DeepUrfold-0.0.1:
                    #oldCoords = torch.sort(labels[0], dim=0)[0]
                #assert knn_points==kdtree_dataloader_points, f"Error: Sapce-filling models are not equal: knn_total={len(knn_points)}; kd_total={len(kdtree_dataloader_points)}; intersection={len(knn_points.intersection(kdtree_dataloader_points))}; knn-kd={len(knn_points-kdtree_dataloader_points)}; kd-knn={len(kdtree_dataloader_points-knn_points)}"
                del labels, knn_points, kdtree_dataloader_points



        z, sout, means, log_vars = self.forward(x) #out_cls, targets, sout, mu, log_var, z

        if visualize_tensors:
            tensors_to_vizualize["SparseTensor output"] = [t.clone().cpu() for t in sout.decomposed_coordinates]

        if self.hparams.tensor_field:
            z = z.slice(tfield)
            sout = sout.slice(tfield)
            means = means.slice(tfield)
            log_vars = log_vars.slice(tfield)

            if visualize_tensors:
                tensors_to_vizualize["TensorField from SparseTensor output"] = [t.clone().cpu() for t in sout.decomposed_coordinates]

        #z, x_hat, p, q
        if self.hparams.return_reconstruction:
            del coords, feats, batch, x, means, tensors_to_vizualize, z
            return sout, {}
        elif hasattr(self.hparams, "samplewise_loss") and self.hparams.samplewise_loss:
            #Try sample wise loss
            print("MEAN IS", means)
            recon_loss, kl, loss, other_features = self.samplewise_loss(
                sout, labels if eval else data, means, log_vars)
            if other_features is not None:
                a = torch.cat([torch.unique(t) for t in other_features])

                #b = torch.unique(torch.cat(other_features))
                #logs = {"other_features": a}
                logs = {}
            else:
                logs = {}
        else:
            #Use regular batchwise loss
            recon_loss, kl, loss, other_features = self.loss(
                sout.F, labels.F if eval else x.F, means.F, log_vars.F)
            logs = {
                "recon_loss": recon_loss,
                "kl": kl,
                "loss": loss,
            }
            if other_features is not None:
                a = torch.cat([torch.unique(t) for t in other_features])
                #b = torch.unique(torch.cat(other_features))
                #logs["other_features"] = a

        if visualize_tensors:
            self.visualize_tensors(batch_idx=batch_idx, **tensors_to_vizualize)

        del coords, feats, batch, x, means, tensors_to_vizualize

        if self.on_gpu and self.global_step % 10 == 0:
            torch.cuda.empty_cache()

        if self.hparams.return_latent and self.hparams.return_var:
            return (z, log_vars), logs
        if self.hparams.return_latent:
            return z, logs
        elif self.hparams.return_reconstruction:
            return sout, logs

        del z

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()},
            on_step=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"val_{k}": v for k, v in logs.items()},
            prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx, eval=True)

        try:
            self.log_dict(
                {f"test_{k}": v for k, v in logs.items()},
                prog_bar=True, logger=True, sync_dist=True)
        except Exception:
            pass
        return loss

    def visualize_tensors(self, *tensors, batch_idx=0, **named_tensors):
        def PointCloud(points, colors=None):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            return pcd

        for i, tensor in enumerate(tensors):
            name = f"Tensor {i}"
            assert name not in named_tensors
            named_tensors[name] = tensor

        for name, batch_coords in named_tensors.items():
            for b, coords in enumerate(batch_coords):
                #pcd = PointCloud(coords.cpu())
                #o3d.visualization.draw_geometries([pcd], window_name=f"{name} (Sample {b})")
                torch.save(coords, f'{name.replace(" ", "_")}_(Batch{batch_idx}Sample{b})')
        assert 0


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

        return [optimizer], [scheduler]

    def configure_sync_batchnorm(self, model):
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        return model

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Make sure batches do not get sent to GPU immediatly if KD-Tree occurs
        in the forward step"""
        if not self.hparams.expand_surface_data_loader and (self.hparams.space_fill_algorithm=="kdtree" or self.hparams.space_fill_algorithm.endswith("cpu")):
            #Keep on CPU, will transfer later
            return batch
        else:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--prefix", default="")

        parser.add_argument('--lr', default=.2, type=float) #0.01
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)

        parser.add_argument('--kernel_size', default=3, type=int)
        parser.add_argument('--transpose_kernel_size', default=2, type=int)
        parser.add_argument('--stride', default=2, type=int)

        parser.add_argument('--sample_during_eval', type=str2bool, nargs='?',
                           const=True, default=False)
        parser.add_argument('--return_latent', type=str2bool, nargs='?',
                           const=True, default=False)
        parser.add_argument('--return_reconstruction', type=str2bool, nargs='?',
                           const=True, default=False)
        parser.add_argument('--return_var', type=str2bool, nargs='?',
                           const=True, default=False)
        parser.add_argument('--tensor_field', type=str2bool, nargs='?',
                           const=True, default=False)

        parser.add_argument("--sweep", type=str2bool, nargs='?',
                           const=True, default=False,
                           help="If running a sweep, tells data loader to use 2 fewer cores")

        return parser
