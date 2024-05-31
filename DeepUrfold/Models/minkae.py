# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch
import torch.nn as nn
import torch.optim as optim

import MinkowskiEngine as ME
from MinkowskiEngineBackend._C import CoordinateMapKey

class Encoder(nn.Module):

    CHANNELS = [16, 32, 64, 128, 256, 512, 1024]

    def __init__(self, nIn, kernel_size=3, stride=2):
        nn.Module.__init__(self)

        # Input sparse tensor must have tensor stride 128.
        ch = self.CHANNELS

        # Block 1
        self.block1 = nn.Sequential(
            ME.MinkowskiConvolution(
                nIn, ch[0], kernel_size=kernel_size, stride=stride, dimension=3),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[0], ch[0], kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
        )

        self.block2 = nn.Sequential(
            ME.MinkowskiConvolution(
                ch[0], ch[1], kernel_size=kernel_size, stride=stride, dimension=3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
        )

        self.block3 = nn.Sequential(
            ME.MinkowskiConvolution(
                ch[1], ch[2], kernel_size=kernel_size, stride=stride, dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[2], ch[2], kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
        )

        self.block4 = nn.Sequential(
            ME.MinkowskiConvolution(
                ch[2], ch[3], kernel_size=kernel_size, stride=stride, dimension=3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[3], ch[3], kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
        )

        self.block5 = nn.Sequential(
            ME.MinkowskiConvolution(
                ch[3], ch[4], kernel_size=kernel_size, stride=stride, dimension=3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
        )

        # Block 5
        self.block6 = nn.Sequential(
            ME.MinkowskiConvolution(
                ch[4], ch[5], kernel_size=kernel_size, stride=stride, dimension=3),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
        )

        # Block 6
        self.block7 = nn.Sequential(
            ME.MinkowskiConvolution(
                ch[5], ch[6], kernel_size=kernel_size, stride=stride, dimension=3),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[6], ch[6], kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
        )

        if ME.__version__ == "0.4.3":
            self.global_pool = ME.MinkowskiGlobalPooling()
        else:
            from MinkowskiEngineBackend._C import PoolingMode
            self.global_pool = ME.MinkowskiGlobalPooling() #PoolingMode.GLOBAL_AVG_POOLING_KERNEL)

        self.embedding = ME.MinkowskiLinear(ch[6], ch[6], bias=True)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, sinput):
        out = self.block1(sinput)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.global_pool(out)
        out = self.embedding(out)
        return out


class Decoder(nn.Module):

    CHANNELS = [1024, 512, 256, 128, 64, 32, 16]
    resolution = 128

    def __init__(self, nOut, kernel_size=3, transpose_kernel_size=2, stride=2):
        nn.Module.__init__(self)

        # Input sparse tensor must have tensor stride 128.
        ch = self.CHANNELS

        # Block 1
        self.block1 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[0],
                ch[0],
                kernel_size=transpose_kernel_size,
                stride=stride,
                dimension=3),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[0], ch[0], kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolutionTranspose(
                ch[0],
                ch[1],
                kernel_size=transpose_kernel_size,
                stride=stride,
                dimension=3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
        )

        # Block 2
        self.block2 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[1],
                ch[2],
                kernel_size=transpose_kernel_size,
                stride=stride,
                dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[2], ch[2], kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
        )

        # Block 3
        self.block3 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[2],
                ch[3],
                kernel_size=transpose_kernel_size,
                stride=stride,
                dimension=3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[3], ch[3], kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
        )

        # Block 4
        self.block4 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[3],
                ch[4],
                kernel_size=transpose_kernel_size,
                stride=stride,
                dimension=3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
        )

        # Block 5
        self.block5 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[4],
                ch[5],
                kernel_size=transpose_kernel_size,
                stride=stride,
                dimension=3),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
        )

        # Block 6
        self.block6 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[5],
                ch[6],
                kernel_size=transpose_kernel_size,
                stride=stride,
                dimension=3),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[6], ch[6], kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
        )

        if ME.__version__ == "0.4.3":
            self.block7 = ME.MinkowskiConvolution(
                ch[6], nOut, kernel_size=1, has_bias=True, dimension=3)
        else:
            self.block7 = ME.MinkowskiConvolution(
                ch[6], nOut, kernel_size=1, bias=True, dimension=3)

        # pruning
        self.pruning = ME.MinkowskiPruning()

    def get_batch_indices(self, out):
        return out.coords_man.get_row_indices_per_batch(out.coords_key)

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool)
            cm = out.coords_man
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0], force_creation=True)
            ins, outs = cm.get_kernel_map(
                out.coords_key,
                strided_target_key,
                kernel_size=2,
                region_type=1)
            for curr_in in ins:
                target[curr_in] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, z):
        out_cls, targets = [], []

        z1 = ME.SparseTensor(
            features=z.F.contiguous(),
            coordinates=z.C,
            tensor_stride=self.resolution,
            coordinate_manager=z.coordinate_manager,
            device=z.F.device)

        # Block1
        out1 = self.block1(z1)
        # target = self.get_target(out1, target_key)
        # targets.append(target)
        # out_cls.append(out1_cls)
        # keep1 = (out1_cls.F > 0).cpu().squeeze()
        #
        # # If training, force target shape generation, use net.eval() to disable
        # if self.training:
        #     keep1 += target
        #
        # # Remove voxels 32
        # out1 = self.pruning(out1, keep1.cpu())

        # Block 2
        out2 = self.block2(out1)
        # out2_cls = self.block2_cls(out2)
        # target = self.get_target(out2, target_key)
        # targets.append(target)
        # out_cls.append(out2_cls)
        # keep2 = (out2_cls.F > 0).cpu().squeeze()
        #
        # if self.training:
        #     keep2 += target
        #
        # # Remove voxels 16
        # out2 = self.pruning(out2, keep2.cpu())

        # Block 3
        out3 = self.block3(out2)
        # out3_cls = self.block3_cls(out3)
        # target = self.get_target(out3, target_key)
        # targets.append(target)
        # out_cls.append(out3_cls)
        # keep3 = (out3_cls.F > 0).cpu().squeeze()
        #
        # if self.training:
        #     keep3 += target
        #
        # # Remove voxels 8
        # out3 = self.pruning(out3, keep3.cpu())

        # Block 4
        out4 = self.block4(out3)
        # out4_cls = self.block4_cls(out4)
        # target = self.get_target(out4, target_key)
        # targets.append(target)
        # out_cls.append(out4_cls)
        # keep4 = (out4_cls.F > 0).cpu().squeeze()
        #
        # if self.training:
        #     keep4 += target
        #
        # # Remove voxels 4
        # out4 = self.pruning(out4, keep4.cpu())

        # Block 5
        out5 = self.block5(out4)
        # out5_cls = self.block5_cls(out5)
        # target = self.get_target(out5, target_key)
        # targets.append(target)
        # out_cls.append(out5_cls)
        # keep5 = (out5_cls.F > 0).cpu().squeeze()
        #
        # if self.training:
        #     keep5 += target
        #
        # # Remove voxels 2
        # out5 = self.pruning(out5, keep5.cpu())

        # Block 5
        out6 = self.block6(out5)
        # out6_cls = self.block6_cls(out6)
        # target = self.get_target(out6, target_key)
        # targets.append(target)
        # out_cls.append(out6_cls)
        # keep6 = (out6_cls.F > 0).cpu().squeeze()

        # Last layer does not require keep
        # if self.training:
        #   keep6 += target

        # Remove voxels 1
        # if keep6.sum() > 0:
        #     out6 = self.pruning(out6, keep6.cpu())

        #return out_cls, targets, out6

        out7 = self.block7(out6)

        return out7


class VAE(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, sinput, gt_target):
        means, log_vars = self.encoder(sinput)
        zs = means
        if self.training:
            zs = zs + torch.exp(0.5 * log_vars.F) * torch.randn_like(log_vars.F)
        out_cls, targets, sout = self.decoder(zs, gt_target)
        return out_cls, targets, sout, means, log_vars, zs


def train(net, dataloader, device, config):
    import logging
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    crit = nn.BCEWithLogitsLoss()

    start_iter = 0
    if config.resume is not None:
        checkpoint = torch.load(config.resume)
        print('Resuming weights')
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_iter = checkpoint['curr_iter']

    net.train()
    train_iter = iter(dataloader)
    # val_iter = iter(val_dataloader)
    logging.info(f'LR: {scheduler.get_lr()}')
    for i in range(start_iter, config.max_iter):

        s = time()
        data_dict = train_iter.next()
        d = time() - s

        optimizer.zero_grad()
        sin = ME.SparseTensor(
            torch.ones(len(data_dict['coords']), 1),
            data_dict['coords'].int(),
            allow_duplicate_coords=True,  # for classification, it doesn't matter
        ).to(device)

        # Generate target sparse tensor
        target_key = sin.coords_key

        out_cls, targets, sout, means, log_vars, zs = net(sin, target_key)
        num_layers, BCE = len(out_cls), 0
        losses = []
        for out_cl, target in zip(out_cls, targets):
            curr_loss = crit(out_cl.F.squeeze(),
                             target.type(out_cl.F.dtype).to(device))
            losses.append(curr_loss.item())
            BCE += curr_loss / num_layers

        KLD = -0.5 * torch.mean(
            torch.mean(1 + log_vars.F - means.F.pow(2) - log_vars.F.exp(), 1))
        loss = KLD + BCE

        loss.backward()
        optimizer.step()
        t = time() - s

        if i % config.stat_freq == 0:
            logging.info(
                f'Iter: {i}, Loss: {loss.item():.3e}, Depths: {len(out_cls)} Data Loading Time: {d:.3e}, Tot Time: {t:.3e}'
            )

        if i % config.val_freq == 0 and i > 0:
            torch.save(
                {
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'curr_iter': i,
                }, config.weights)

            scheduler.step()
            logging.info(f'LR: {scheduler.get_lr()}')

            net.train()


def visualize(net, dataloader, device, config):
    net.eval()
    crit = nn.BCEWithLogitsLoss()
    n_vis = 0

    for data_dict in dataloader:

        sin = ME.SparseTensor(
            torch.ones(len(data_dict['coords']), 1),
            data_dict['coords'].int(),
            allow_duplicate_coords=True,  # for classification, it doesn't matter
        ).to(device)

        # Generate target sparse tensor
        target_key = sin.coords_key

        out_cls, targets, sout, means, log_vars, zs = net(sin, target_key)
        num_layers, BCE = len(out_cls), 0
        losses = []
        for out_cl, target in zip(out_cls, targets):
            curr_loss = crit(out_cl.F.squeeze(),
                             target.type(out_cl.F.dtype).to(device))
            losses.append(curr_loss.item())
            BCE += curr_loss / num_layers

        KLD = -0.5 * torch.mean(
            torch.sum(1 + log_vars.F - means.F.pow(2) - log_vars.F.exp(), 1))
        loss = KLD + BCE

        print(loss)

        batch_coords, batch_feats = sout.decomposed_coordinates_and_features
        for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
            pcd = PointCloud(coords)
            pcd.estimate_normals()
            pcd.translate([0.6 * config.resolution, 0, 0])
            pcd.rotate(M)
            opcd = PointCloud(data_dict['xyzs'][b])
            opcd.translate([-0.6 * config.resolution, 0, 0])
            opcd.estimate_normals()
            opcd.rotate(M)
            o3d.visualization.draw_geometries([pcd, opcd])

            n_vis += 1
            if n_vis > config.max_visualization:
                return


if __name__ == '__main__':
    config = parser.parse_args()
    logging.info(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = VAE()
    net.to(device)

    logging.info(net)

    if config.train:
        dataloader = make_data_loader(
            'train',
            augment_data=True,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            repeat=True,
            config=config)

        train(net, dataloader, device, config)
    else:
        import os
        if not os.path.exists(config.weights):
            logging.info(
                f'Downloaing pretrained weights. This might take a while...')
            urllib.request.urlretrieve(
                "https://bit.ly/39TvWys", filename=config.weights)

        logging.info(f'Loading weights from {config.weights}')
        checkpoint = torch.load(config.weights)
        net.load_state_dict(checkpoint['state_dict'])

        dataloader = make_data_loader(
            'test',
            augment_data=True,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            repeat=True,
            config=config)

        with torch.no_grad():
            visualize(net, dataloader, device, config)
