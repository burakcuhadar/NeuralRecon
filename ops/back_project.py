import torch
from torch.nn.functional import grid_sample
import math

def back_project(coords, origin, voxel_size, feats, KRcam, use_sparse_method1=False, depth_im=None):
    '''
    Unproject the image fetures to form a 3D (sparse) feature volume

    :param coords: coordinates of voxels,
    dim: (num of voxels, 4) (4 : batch ind, x, y, z)
    :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
    dim: (batch size, 3) (3: x, y, z)
    :param voxel_size: floats specifying the size of a voxel
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix
    dim: (num of views, batch size, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (num of voxels, c + 1)
    :return: count: number of times each voxel can be seen
    dim: (num of voxels,)
    '''
    n_views, bs, c, h, w = feats.shape

    if use_sparse_method1:
        feature_volume_all = torch.zeros(coords.shape[0], c + 2).cuda()
    else:
        feature_volume_all = torch.zeros(coords.shape[0], c + 1).cuda()
    count = torch.zeros(coords.shape[0]).cuda()

    for batch in range(bs):
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
        coords_batch = coords[batch_ind][:, 1:]

        coords_batch = coords_batch.view(-1, 3)
        origin_batch = origin[batch].unsqueeze(0)
        feats_batch = feats[:, batch]
        proj_batch = KRcam[:, batch]

        grid_batch = coords_batch * voxel_size + origin_batch.float()
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        nV = rs_grid.shape[-1]
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1)

        # Project grid
        im_p = proj_batch @ rs_grid
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z


        # Find nearest depth value for each voxel
        if use_sparse_method1:
            sparse_depth_feature = torch.zeros(n_views,nV).cuda()
            for view in range(n_views):
                depth = depth_im[view,batch] #(h,w)

                # find the nearest depth value for every projected voxel
                xx, yy = torch.meshgrid(torch.arange(h, device=torch.device('cuda')), torch.arange(w, device=torch.device('cuda')))
                xx = xx.reshape(h*w)
                yy = yy.reshape(h*w)        # (h*w)

                depth_mask = depth.view(h*w) > 0
                depth_coords = torch.stack([xx[depth_mask], yy[depth_mask]], dim=-1) # (num of valid depths, 2)
                #depth_coords = depth_coords.unsqueeze(0).expand(nV, -1, 2)           # (num voxels, num of valid depths, 2)
                #depth_coords = depth_coords.cuda()
                im_coords = torch.stack([im_x[view], im_y[view]], dim=-1)         # (num voxels,  2)
                #im_coords = im_coords.unsqueeze(1).expand(nV, depth_mask.sum(),2) # (num voxels, num of valid depths,2)

                #dist = im_coords - depth_coords                                   #  num voxels, num depth, 2)
                #dist = dist.square().sum(dim=-1).sqrt()                           # (num voxels, num depth)
                dist = torch.cdist(im_coords, depth_coords)
                nearest_idx = torch.argmin(dist, dim=-1)                          # (num voxels)
                # For each voxel in each view "nearest_dist" stores the distance to the nearest pixel that has valid depth
                nearest_dist = dist[torch.arange(nV),nearest_idx] # (num voxels)
                # Normalize nearest_dist into [0,1]
                nearest_dist = nearest_dist / (h**2 + w**2)**0.5
                # For each voxel in each view "nearest_depth" stores the depth of the nearest neighbor that has valid depth
                #depth = depth.view(h,w).unsqueeze(0).expand(nV, -1, -1).view(nV, -1) # (num voxels, h*w)
                #nearest_x = depth_coords[:,nearest_idx,0]
                #nearest_y = depth_coords[:,nearest_idx,1]             #  (num voxels)
                #nearest_depth = depth[nearest_x, nearest_y]           #  (num voxels)
                nearest_depth = depth.view(h*w)[depth_mask].unsqueeze(0).expand(nV,-1)[torch.arange(nV), nearest_idx]
                # create the additional feature dimension
                sparse_depth_feature[view] = torch.where(torch.abs(im_z[view] - nearest_depth) < voxel_size * math.sqrt(2.) / 2., 1 - nearest_dist, torch.zeros(nV, device=torch.device('cuda')))


        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        feats_batch = feats_batch.view(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2)
        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)

        features = features.view(n_views, c, -1)
        mask = mask.view(n_views, -1)
        im_z = im_z.view(n_views, -1)
        # remove nan
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        im_z[mask == False] = 0

        if use_sparse_method1:
            sparse_depth_feature[mask == False] = 0
            sparse_depth_feature = sparse_depth_feature.unsqueeze(1)
            features = torch.cat([features, sparse_depth_feature], dim=1)
            

        count[batch_ind] = mask.sum(dim=0).float()

            
        # aggregate multi view
        features = features.sum(dim=0)
        mask = mask.sum(dim=0)
        invalid_mask = mask == 0
        mask[invalid_mask] = 1
        in_scope_mask = mask.unsqueeze(0)
        features /= in_scope_mask
        features = features.permute(1, 0).contiguous()

        # concat normalized depth value
        im_z = im_z.sum(dim=0).unsqueeze(1) / in_scope_mask.permute(1, 0).contiguous()
        im_z_mean = im_z[im_z > 0].mean()
        im_z_std = torch.norm(im_z[im_z > 0] - im_z_mean) + 1e-5
        im_z_norm = (im_z - im_z_mean) / im_z_std
        im_z_norm[im_z <= 0] = 0
        features = torch.cat([features, im_z_norm], dim=1)

        feature_volume_all[batch_ind] = features
    return feature_volume_all, count
