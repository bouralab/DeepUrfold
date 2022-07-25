import math
from collections import defaultdict

import torch
import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME

from molmimic.common.ProteinTables import vdw_radii, vdw_r_volume
from molmimic.common.features import atom_features, atom_feature_aggregegation

search_algorithms = {}

aggregegation_fns_map = {
    "max": lambda t: torch.max(t, dim=0)[0],
    "min": lambda t: torch.min(t, dim=0)[0],
    "mean": lambda t: torch.mean(t, dim=0)
}

class VanDerWallsSurface(nn.Module):
    def __init__(self, volume=256, voxel_size=1, algorithm="faiss-cpu", features=None, aggregate=None, device=None):
        assert algorithm in search_algorithms.keys() #["frnn", "faiss-cpu", "faiss-gpu", "faiss", "torch-cluster"]
        super().__init__()
        self.volume = volume #nn.Parameter(torch.Tensor(volume), requires_grad=False)
        self.voxel_size = voxel_size #nn.Parameter(torch.Tensor(voxel_size), requires_grad=False)
        self.grid = None #nn.Parameter(requires_grad=False)
        self.grid_params = None #nn.ParameterList([])
        self.algorithm = search_algorithms[algorithm] #nn.Parameter(algorithm)
        self.device = device

        if aggregate is not None:
            self.aggregegation_fns = [aggregegation_fns_map.get(agg, agg) \
                for agg in aggregate]
        else:
            if features is None:
                features = atom_features
            self.aggregegation_fns = [ \
                aggregegation_fns_map[atom_feature_aggregegation[f]] for f in features]

    def forward(self, coordinates_radii, features, other=None, device=None):
        assert features.size()[-1]==len(self.aggregegation_fns), (features.size(), len(self.aggregegation_fns))

        if other is not None:
            _other = other
            other = [[sorted(i[~torch.all(torch.isnan(i),dim=1)].int().tolist()) \
                for i in o] for o in other]

        if device is None:
            device = features.device if self.device is None else self.device

        coordinates, radii = coordinates_radii[:, :, :3], coordinates_radii[:, :, 3]
        radii = (torch.round(radii*10**3)/10**3)

        if self.grid is None:
            self.grid = self.make_grid(self.volume, self.voxel_size,
                device=device)

        coord_batches = [torch.empty(0, 4, device=device) \
            for batch in range(coordinates.size()[0])]
        feat_batches = [torch.empty(0, features.size()[-1], device=device) \
            for batch in range(coordinates.size()[0])]

        atom2voxels = [[None]*len(i[~torch.all(torch.isnan(i),dim=1)]) for i in coordinates_radii[:,:,:3]]

        unique_radii = radii.unique()
        unique_radii, _ = torch.sort(unique_radii[~torch.isnan(unique_radii)], descending=True)

        for r in unique_radii:
            k = 36 #int(math.ceil(vdw_r_volume[r.item()]))+1
            atom_idx = torch.where(radii==r.item())
            chunk_sizes = torch.unique_consecutive(atom_idx[0], return_counts=True)[1].tolist()

            #Create one point cloud with all data points, [N*n_prots, 3]
            coords = coordinates[atom_idx].float()
            feats = features[atom_idx]
            feats = feats[~torch.all(feats.isnan(),dim=1)]
            length1 = torch.LongTensor([coords.size()[0]]).to(device)
            length2 = torch.LongTensor([self.grid.size()[0]]).to(device)

            result = self.algorithm(
                coords,
                self.grid,
                length1,
                length2,
                chunk_sizes,
                feats,
                r,
                self.grid_params
            )
            coord_feat_chunks, grid_params_ = result[:2]
            if len(result) > 2:
                raw_output = result[2]
            else:
                raw_output = None
            if self.grid_params is None:
                self.grid_params = grid_params_

            #import pdb; pdb.set_trace()

            for batch_index, (coord_chunk, feat_chunk) in enumerate(coord_feat_chunks):
                #Add Batch number in first column
                batch_col = torch.ones(len(coord_chunk), 1, device=coord_chunk.device)*batch_index
                coord_chunk = torch.cat((batch_col, coord_chunk), dim=1).long()
                coord_batches[batch_index] = torch.cat((coord_batches[batch_index], coord_chunk.to(device)))
                feat_batches[batch_index] = torch.cat((feat_batches[batch_index], feat_chunk.to(device)))

            if raw_output is not None:
                for batch_idx, _atom_idx, atom_voxels in zip(*atom_idx, raw_output):
                    atom2voxels[batch_idx][_atom_idx] = sorted(atom_voxels.int().tolist())

        if not all([o==f for o, f in zip(other,atom2voxels)]):
            import pdb; pdb.set_trace()
            #len([(i,o,f) for i,(o, f) in enumerate(zip(other[8],atom2voxels[8])) if o!=f]
            #[i for i,(r,c) in enumerate(zip(*atom_idx)) if r==2 and c==diff[0][0]]
            #pts = self.grid_params.query_ball_point(bbb, r=1.2)


        all_coordinates = torch.cat(coord_batches)
        all_features = torch.cat(feat_batches)

        unique_coords = torch.unique(all_coordinates, dim=0, sorted=True)

        # features = torch.stack([features[torch.where(coordinates==u)[0]].max(0).values \
        #     for u in unique_coords])

        volume_features = torch.empty(len(unique_coords), all_features.size()[-1], device=device)
        for i, u in enumerate(unique_coords):
            features_at_unique_coord = all_features[torch.where((all_coordinates==u).all(dim=1))[0]]
            agg_feats = [agg(features_at_unique_coord[:,i]) \
                for i, agg in enumerate(self.aggregegation_fns)]
            agg_features_at_unique_coord = torch.stack(agg_feats)
            volume_features[i] = agg_features_at_unique_coord

        #import pdb; pdb.set_trace()

        #del atom2voxels, other

        return ME.SparseTensor(
            volume_features,
            coordinates=unique_coords.int(),
            device=device), atom2voxels

    def forward2(self, coordinates_radii, features, lengths):
        assert features.size()[-1]==len(self.aggregegation_fns)

        coordinates, radii = coordinates_radii[:, :, :3], coordinates_radii[:, :, 3]

        if self.grid is None:
            self.grid = self.make_grid(self.volume, self.voxel_size,
                device=coordinates_radius.device)

        idx, grid_params = self.algorithm(self.grid, coordinates, radii, features, lengths, K=32,
            grid_params=self.grid_params if len(self.grid_params)>0 else None)
        self.grid_params = nn.ParameterList(grid_params, requires_grad=False)

        coordinates = self.grid[idx]

        unique_coords = torch.unique(coordinates, sorted=True)
        # perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        # inverse, perm = inverse.flip([0]), perm.flip([0])
        # perm = inverse.new_empty(unique_coords.size(0)).scatter_(0, inverse, perm)

        features = torch.stack([features[torch.where(coordinates==u)[0]].max(0).values \
            for u in unique_coords])

        return ME.SparseTensor(features, coordinates=unique_coords)

    @staticmethod
    def make_grid(volume, voxel_size, batch_size=1, device=None, coordinates=None):
        device = device or "cpu"

        if coordinates is not None:
            min_coord = coordinates[:, :3].min(0).floor()-5
            max_coord = coordinates[:, :3].max(0).ceil()+5
        else:
            min_coord = [0]*3
            max_coord = [volume]*3

        extent_x = torch.arange(min_coord[0], max_coord[0], voxel_size, device=device)
        extent_y = torch.arange(min_coord[1], max_coord[1], voxel_size, device=device)
        extent_z = torch.arange(min_coord[2], max_coord[2], voxel_size, device=device)
        mx, my, mz = torch.meshgrid(extent_x, extent_y, extent_z)

        grid = torch.stack((
            mx.contiguous().view(-1),
            my.contiguous().view(-1),
            mz.contiguous().view(-1)), dim=1).to(device).float()

        del extent_x, extent_y, extent_z, mx, my, mz, min_coord, max_coord

        if batch_size > 1:
            grid = grid.expand(batch_size,-1,-1)

        return grid

def sphere_volume(r):
    return int(math.ceil((4/3)*3.14*(r**3)))

def frnn_gpu(coords, grid, length1, length2, chunk_sizes, features, r, grid_params):
    import frnn
    # first time there is no cached grid

    # device = features.device
    #
    # coord_batches = [torch.empty(0, 4, device=device) \
    #     for batch in range(coordinates.size()[0])]
    # feat_batches = [torch.empty(0, features.size()[-1], device=device) \
    #     for batch in range(coordinates.size()[0])]
    #
    # for r in radii.unique():
    #     k = int(math.ceil(vdw_r_volume[r.item()]))
    #     atom_idx = torch.where(radii==r.item())
    #
    #     #Create one point cloud with all data points, [N*n_prots, 3]
    #     coords = coordinates[atom_idx].unsqueeze(0).float()
    #     feats = features[atom_idx].unsqueeze(0)
    #     length1 = torch.LongTensor([coords.size()[1]], devive=device)
    #     length2 = torch.LongTensor([grid.size()[1]], devive=device)

    k=31 #Cannot be >=32, but some are so this doesn;t work

    print("frnn")

    dists, idxs, _, grid_params_ = frnn.frnn_grid_points(
        coords.unsqueeze(0),
        grid.unsqueeze(0),
        length1,
        length2,
        k,
        r.unsqueeze(0)+.0001,
        grid=grid_params,
        return_nn=False,
        return_sorted=True
    )
    idxs = idxs[0]
    idx_mask = (idxs!=-1)
    new_idx = idxs[idx_mask]
    atm_cnt = idx_mask.int().sum(dim=1)

    atom_volumes = grid[new_idx]

    #feats = feats[0]
    expanded_feats = features.repeat(1,k).view(features.size()[0]*k, features.size()[-1])
    expanded_feats = expanded_feats[idx_mask.view(-1)]

    #chunk_sizes = torch.unique_consecutive(atom_idx[0], return_counts=True)[1].tolist()

    idx_chunks = [idx_chunk.sum() for idx_chunk in torch.split(atm_cnt, chunk_sizes)]

    coord_chunks = torch.split(atom_volumes, idx_chunks)
    feat_chunks = torch.split(expanded_feats, idx_chunks)

    return zip(coord_chunks, feat_chunks), grid_params_, [grid[idxs[i,a]] for i, a in enumerate(idx_mask)]

    return idxs, atm_cnt, grid_params #nn.ParameterList(grid_params, requires_grad=False)

    #     idxs = idxs[0]
    #     idx_mask = (idxs!=-1)
    #     new_idx = idxs[idx_mask]
    #
    #     atom_volumes = grid[new_idx]
    #
    #     feats = feats[0]
    #     expanded_feats = feats.repeat(1,k).view(feats.size()[0]*k, feats.size()[-1])
    #     expanded_feats = expanded_feats[idx_mask.view(-1)]
    #
    #     chunk_sizes = torch.unique_consecutive(atom_idx[0], return_counts=True)[1].tolist()
    #     atm_cnt = idx_mask.int().sum(axis=1)
    #     idx_chunks = [idx_chunk.sum() for idx_chunk in torch.split(atm_cnt, chunk_sizes)]
    #
    #     coord_chunks = torch.split(atom_volume, idx_chunks)
    #     feat_chunks = torch.split(expanded_feats, idx_chunks)
    #
    #     for batch_index, (coord_chunk, feat_chunk) in enumerate(zip(coord_chunks, feat_chunks)):
    #         #Add Batch number in first column
    #         batch_col = torch.ones(idx_chunks[batch_index], 1, device=device)*batch_index
    #         coord_chunk = torch.cat((batch_col, coord_chunk), dim=1).long()
    #         coord_batches[batch_index] = torch.cat((coord_batches[batch_index], coord_chunk))
    #         feat_batches[batch_index] = torch.cat((feat_batches[batch_index], feat_chunk))
    #
    # coordinates = torch.cat(coord_batches)
    # features = torch.cat(feat_batches)
    #
    # unique_coords = torch.unique(coordinates, sorted=True)
    #
    # # features = torch.stack([features[torch.where(coordinates==u)[0]].max(0).values \
    # #     for u in unique_coords])
    #
    # features = torch.Tensor()
    # for u in unique_coords:
    #     features_at_unique_coord = features[torch.where(coordinates==u)[0]]
    #     agg_features_at_unique_coord = torch.cat((agg(features_at_unique_coord[i]) \
    #         for i, agg in enumerate(self.aggregegation_fns)), dim=1)
    #     features = torch.cat((features, agg_features_at_unique_coord), dim=0)
    #
    # return ME.SparseTensor(features, coordinates=unique_coords, device=device)

search_algorithms["frnn"] = frnn_gpu
search_algorithms["frnn-gpu"] = frnn_gpu

def pytorch_cluster(coords, grid, length1, length2, chunk_sizes, features, r, grid_params):
    from torch_cluster import radius
    feature_index, grid_index = radius(x=coords, y=grid, r=r)
    return grid_index, []
search_algorithms["pytorch-cluster"] = pytorch_cluster

def pytorch3d_cpu(coords, grid, length1, length2, chunk_sizes, features, r, grid_params):
    _, idxs, _ = knn_points(
        grid, coords, length1, length2, k, version=-1, return_nn=False, return_sorted=True
    )
    return idxs, []

def faiss_cpu(coords, grid, length1, length2, chunk_sizes, features, r, index=None):
    import faiss
    import faiss.contrib.torch_utils

    if index is None:
        # Add to CPU index with np
        index = faiss.IndexFlatL2(3)
        index.add(grid.cpu())

    lims, D, I = radius_search(index, coords.cpu(), r.item(),
        init_k=100, max_k=2048, gpu=False)

    n_feats = features.size()[-1]
    # chunks = (
    #     (grid[I[lims[i]:lims[i+1]]],
    #     features[i].repeat(1,lims[i+1]-lims[i]).view(
    #         features[i].size()[0]*(lims[i+1]-lims[i]), n_feats)) \
    #     for i in range(len(lims)-1))
    # return chunks, index
    limits = torch.stack((lims[:-1], lims[1:]), dim=1)
    limit_chunks = torch.split(limits, chunk_sizes)
    feat_idx = 0

    batch_coords = []
    batch_feats = []
    for limit_chunk in limit_chunks:
        chunk_coords = []
        chunk_feats = []
        for start, end in limit_chunk:

            # if start<len(lims):
            #     start = lims[start]
            # else:
            #     import pdb; pdb.set_trace()
            #
            # if end<len(lims):
            #     end = lims[end]
            # else:
            #     end = lims[-1]
            #
            # print(start, end, len(lims))
            n_coords = end-start
            volume_coords = grid[I[start:end]]

            feats = features[feat_idx]

            try:
                feats = feats.repeat(1, n_coords).view(n_coords.item(), n_feats)
            except:
                import pdb; pdb.set_trace()
            chunk_coords.append(volume_coords)
            chunk_feats.append(feats)

            feat_idx += 1

        assert len(chunk_coords)>0 and chunk_coords.count(None)==0

        batch_coords.append(torch.cat(chunk_coords))
        batch_feats.append(torch.cat(chunk_feats))

    #kdtree_chunks, kdindex, _ = KDTree_cpu(coords, grid, length1, length2, chunk_sizes, features, r, index=kdindex)

    return zip(batch_coords, batch_feats), index, [grid[I[lims[i].item():lims[i+1].item()]] for i in range(len(lims)-1)]#[index, kdindex]


    # for r in radii.unique()[0]:
    #     k = vdw_r_volume[r]
    #     atom_idx = torch.where(radii==r)
    #
    #     #Create one point cloud with all data points, [N*n_prots, 3]
    #     coords = coordinates[atom_idx]
    #     feats = features[atom_idx]
    #
    #     lims, I, D = radius_search(index, coords, r,
    #         max_k=k, init_k=k, gpu=False)
    #
    #     lims, I, D = radius_search(index, coords, r,
    #         max_k=k, init_k=k, gpu=False)

    idx = [I[lims[i]:lims[i+1]] for i in range(len(lims)-1)]
    return idx, index
search_algorithms["faiss-cpu"] = faiss_cpu

def faiss_gpu(coords, grid, length1, length2, chunk_sizes, features, r, index=None):
    import faiss
    import faiss.contrib.torch_utils

    if index is None:
        # Add to GPU with torch GPU
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = coords.device.index
        index = faiss.GpuIndexFlatL2(res, 3, flat_config)
        index.add(grid)
        #kdindex=None
    #else:
    #    index, kdindex = index

    lims, D, I = radius_search(index, coords, (r+.00001).pow(2).item(),
        init_k=100, max_k=500, gpu=True)

    n_feats = features.size()[-1]
    # chunks = (
    #     (grid[I[lims[i]:lims[i+1]]],
    #     features[i].repeat(1,lims[i+1]-lims[i]).view(
    #         features[i].size()[0]*(lims[i+1]-lims[i]), n_feats)) \
    #     for i in range(len(lims)-1))
    # return chunks, index
    limits = torch.stack((lims[:-1], lims[1:]), dim=1)
    limit_chunks = torch.split(limits, chunk_sizes)
    feat_idx = 0

    batch_coords = []
    batch_feats = []
    for limit_chunk in limit_chunks:
        chunk_coords = []
        chunk_feats = []
        for start, end in limit_chunk:

            # if start<len(lims):
            #     start = lims[start]
            # else:
            #     import pdb; pdb.set_trace()
            #
            # if end<len(lims):
            #     end = lims[end]
            # else:
            #     end = lims[-1]
            #
            # print(start, end, len(lims))
            n_coords = end-start
            volume_coords = grid[I[start:end]]

            feats = features[feat_idx]

            try:
                feats = feats.repeat(1, n_coords).view(n_coords.item(), n_feats)
            except:
                import pdb; pdb.set_trace()
            chunk_coords.append(volume_coords)
            chunk_feats.append(feats)

            feat_idx += 1

        assert len(chunk_coords)>0 and chunk_coords.count(None)==0

        batch_coords.append(torch.cat(chunk_coords))
        batch_feats.append(torch.cat(chunk_feats))

    #kdtree_chunks, kdindex, _ = KDTree_cpu(coords, grid, length1, length2, chunk_sizes, features, r, index=kdindex)

    return zip(batch_coords, batch_feats), index, [grid[I[lims[i].item():lims[i+1].item()]] for i in range(len(lims)-1)]#[index, kdindex]



    chunks = [[[],[]] for _ in range(chunk_sizes)]
    chunk_idxs = torch.split(torch.arange(len(lims)-1), chunk_sizes)
    #batch_idxs = torch.repeat_interleave(torch.arange(len(lims)-1), torch.Tensor(chunk_sizes), 0)
    last_idx = 0
    for batch_idx, chunk in enumerate(chunk_idxs):
        for i in range(1,len(chunk)):
            coords = grid[I[lims[i]:lims[i+1]]]
            feats = features[i].repeat(1,lims[i+1]-lims[i]).view(
                lims[i+1]-lims[i], n_feats)
            batches[batch_idx][0].append(coords)
            batches[batch_idx][1].append(feats)
    chunks = ((torch.cat(c), torch.cat(f)) for c, f in chunks)
    return chunks

    index = faiss.IndexFlatL2(3)
    index.add(grid.cpu().numpy())

    for r in radii.view(-1).unique():
        k = sphere_volume(r)
        atom_idx = torch.where(radii==r)
        coords = coordinates[atom_idx]
        feats = features[atom_idx]
        lims, I, D = radius_search(index, coords, r,
            max_k=k, init_k=k, gpu=True)

        for i in range(len(lims)-1):
            grid.view(-1)[I[lims[i]:lims[i+1]]].resize((batch_size, 3))

        idx = [I[lims[i]:lims[i+1]] for i in lims]
        grid[idx]


    # D, I = index.search(coordinates[:, :3], 113)
    # mask = (D < coordinates[:, -1].pow(2)).int()
    # torch.where(mask==1)
    #
    # range_search_gpu()


    return idx, []
search_algorithms["faiss-gpu"] = faiss_gpu
search_algorithms["faiss"] = faiss_gpu

def radius_search(index, x, thresh, nprobe=None, max_k=2048, init_k=100, gpu=False, eps=.001):
    """The method range_search returns all vectors within a radius around the
    query point (as opposed to the k nearest ones). Since the result lists for
    each query are of different sizes, it must be handled specially:

        in C++ it returns the results in a pre-allocated RangeSearchResult
        structure

        in Python, the results are returned as a triplet of 1D arrays lims, D,
        I, where result for query i is in I[lims[i]:lims[i+1]] (indices of
        neighbors), D[lims[i]:lims[i+1]] (distances).

    https://github.com/facebookresearch/faiss/issues/565
    """
    if nprobe is not None:
        if hasattr(index, 'nprobe'):
            index.nprobe = nprobe
        else:
            faiss.downcast_index(index).nprobe = nprobe

    if not gpu:
        lims, D, I = index.range_search(x, thresh=thresh)

        # faiss doesn't sort
        for i in range(len(lims)-1):
            sorted_idx = np.argsort(D[lims[i]: lims[i+1]])
            D[lims[i]: lims[i+1]] = D[lims[i]: lims[i+1]][sorted_idx]
            I[lims[i]: lims[i+1]] = I[lims[i]: lims[i+1]][sorted_idx]

    else:
        device = x.device
        ii = defaultdict(list)
        dd = defaultdict(list)

        k = init_k
        D, I = index.search(x, k)
        n = len(D)
        r, c = torch.where(D <= thresh)
        actual_r = r

        while True:
            for row, col, act_r in zip(r, c, actual_r):
                act_r = act_r.item()
                ii[act_r].append(I[row, col])
                dd[act_r].append(D[row, col])

            continue_idx = [rr for rr, v in dd.items() if len(v) == k]

            if len(continue_idx) == 0:
                break

            k *= 2
            if k >= max_k:
                break

            D, I = index.search(x[continue_idx], k=k)

            prev_k = int(k/2)
            D = D[:, prev_k:]
            I = I[:, prev_k:]
            r, c = torch.where(D <= thresh)
            _, cts = torch.unique(r, return_counts=True)
            actual_r = torch.repeat(continue_idx, cts)

        sorted_rows = list(range(n))

        #print(np.cumsum([dd[i].size()[0] for i in sorted_rows]))
        lims = torch.cat([
            torch.IntTensor([0]),
            torch.cumsum(torch.IntTensor([len(dd[i]) for i in sorted_rows]), 0)]
            ).int()
        D = torch.FloatTensor([sl for l in (dd[r] for r in sorted_rows) for sl in l]).to(device)
        I = torch.LongTensor([sl for l in (ii[r] for r in sorted_rows) for sl in l]).to(device)

    return lims, D, I

def range_search_gpu(xq, r2, index_gpu, index_cpu):
    """GPU does not support range search, so we emulate it with
    knn search + fallback to CPU index.
    The index_cpu can either be a CPU index or a numpy table that will
    be used to construct a Flat index if needed.
    """
    nq, d = xq.shape
    LOG.debug("GPU search %d queries" % nq)
    k = min(index_gpu.ntotal, 1024)
    D, I = index_gpu.search(xq, k)
    if index_gpu.metric_type == faiss.METRIC_L2:
        mask = D[:, k - 1] < r2
    else:
        mask = D[:, k - 1] > r2
    if mask.sum() > 0:
        LOG.debug("CPU search remain %d" % mask.sum())
        if isinstance(index_cpu, np.ndarray):
            # then it in fact an array that we have to make flat
            xb = index_cpu
            index_cpu = faiss.IndexFlat(d, index_gpu.metric_type)
            index_cpu.add(xb)
        lim_remain, D_remain, I_remain = index_cpu.range_search(xq[mask], r2)
    LOG.debug("combine")
    D_res, I_res = [], []
    nr = 0
    for i in range(nq):
        if not mask[i]:
            if index_gpu.metric_type == faiss.METRIC_L2:
                nv = (D[i, :] < r2).sum()
            else:
                nv = (D[i, :] > r2).sum()
            D_res.append(D[i, :nv])
            I_res.append(I[i, :nv])
        else:
            l0, l1 = lim_remain[nr], lim_remain[nr + 1]
            D_res.append(D_remain[l0:l1])
            I_res.append(I_remain[l0:l1])
            nr += 1
    lims = np.cumsum([0] + [len(di) for di in D_res])
    return lims, np.hstack(D_res), np.hstack(I_res)

def KDTree_cpu(coords, grid, length1, length2, chunk_sizes, features, r, index=None):
    import more_itertools
    from scipy import spatial

    #device = coords.device
    #coords = coords.cpu()
    #grid = grid.cpu()

    if index is None:
        index = spatial.cKDTree(grid.float().tolist()) #numpy())

    try:
        voxels = index.query_ball_point(np.round(np.array(coords.tolist())*10**4)/10**4, r=r.item())
    except:
        import pdb; pdb.set_trace()

    assert len(voxels) == sum(chunk_sizes)

    try:
        batch_coords = [torch.cat([grid[v] for v in v_chunk]) \
            for v_chunk in more_itertools.split_into(voxels, chunk_sizes)]
    except:
        import pdb; pdb.set_trace()

    batch_feats = [features[i].repeat(1,len(volume_voxels)).view(
        len(volume_voxels), features.size()[1]) for i, volume_voxels in enumerate(voxels)]
    batch_feats = [torch.cat(f) for f in more_itertools.split_into(
        batch_feats, chunk_sizes)]

    assert len(batch_coords)==len(batch_feats)

    return zip(batch_coords, batch_feats), index, [grid[v] for v in voxels]
search_algorithms["kdtree"] = KDTree_cpu
search_algorithms["kdtree-cpu"] = KDTree_cpu


if __name__ == "__main__":
    from DeepUrfold.DataModules.DomainStructureDataModule import collate_coordinates
    from DeepUrfold.Models.van_der_waals_surface import VanDerWallsSurface
    from DeepUrfold.Datasets.DomainStructureDataset import DomainStructureDataset
    from DeepUrfold.Models.van_der_waals_surface import make_grid
    from molmimic.common.ProteinTables import vdw_r_volume
    import torch, math, frnn

    atom_features = [
         'C', 'A', 'N', 'OA', 'OS', 'C_elem', 'N_elem', 'O_elem', 'S_elem',
         'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
         'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR',
         'is_helix', 'is_sheet', 'residue_buried', 'is_hydrophobic', 'pos_charge',
         'is_electronegative'
        ]

    ds = DomainStructureDataset("/home/bournelab/data-eppic-cath-features/train_files/2/60/40/10/DomainStructureDataset-train-C.A.T.H.S35-split0.8.h5", data_dir="/home/bournelab/data-eppic-cath-features", use_features=atom_features, volume=256, expand_surface_data_loader=False)
    data = collate_coordinates([ds[0], ds[1], ds[2], ds[3]])
    coordinates_radii, features, _, lengths = data
    coordinates, radii = coordinates_radii[:, :, :3], coordinates_radii[:, :, 3]

    for r in radii.unique():
        break



    grid_params = None
    coordinates = coordinates.float().cuda()
    features = features.float().cuda()
    grid = make_grid(256, 1, 1).float().unsqueeze(0).cuda()
