import torch.nn.functional as F
from pointndf.backbone.model import PointTransformerV3
import torch
import torch.nn as nn
from pointndf.layers import PointNet_SA_Module_KNN, MLP_CONV, Transformer, ShapeAttention, query_knn
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, ball_query, three_nn, three_interpolate, grouping_operation

class FeatureExtractor(nn.Module):

    def __init__(self, out_dim=256):
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(256, 32, 3, [32, 64], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_2 = PointNet_SA_Module_KNN(64, 32, 64, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [256, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud):
        l0_xyz = point_cloud
        l0_points = point_cloud
        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points, None)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points, None)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points, None)
        return l3_points

def position_encoding(direc, pe_out_L=1):
    trt = direc
    for i in range(pe_out_L):
        trt = torch.cat((trt, torch.sin(direc * 2 ** i * 3.14159265)), dim=-1)
        trt = torch.cat((trt, torch.cos(direc * 2 ** i * 3.14159265)), dim=-1)
    return trt

class DirectProj(nn.Module):

    def __init__(self, dim=128, lsp_dim=64):
        super(DirectProj, self).__init__()
        self.smooth_mlp = MLP_CONV(in_channel=lsp_dim + dim, layer_dims=[128, 128, 64], bn=True)
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[16, 64], bn=True)
        self.skip = MLP_CONV(in_channel=lsp_dim + 64, layer_dims=[64], bn=True)
        self.mlp_3 = MLP_CONV(in_channel=64, layer_dims=[128, lsp_dim], bn=True)
        self.sa = Transformer(lsp_dim, dim=64)
        self.dir_mlp = MLP_CONV(in_channel=64 + 9, layer_dims=[64, 16, 3], bn=True)
        self.dist_mlp = MLP_CONV(in_channel=64 + 9 + 9, layer_dims=[64, 16, 1], bn=True)

    def forward(self, global_shape_fea, parent_pos, parent_lspfea):
        batch = parent_pos.size(0)
        N = parent_pos.size(2)
        shape_fea = global_shape_fea.repeat(1, 1, N)
        feat_1 = self.mlp_1(parent_pos)
        if parent_lspfea != None:
            feat_1 = self.skip(torch.cat([parent_lspfea, feat_1], -2))
        mfea = self.mlp_3(feat_1)
        lsp_fea = self.sa(mfea, parent_pos)
        spatial_fea = self.smooth_mlp(torch.cat([lsp_fea, shape_fea], -2))
        pos = parent_pos.permute(0, 2, 1).contiguous()
        pos_emb = position_encoding(pos.view(-1, 3))
        pos_emb = pos_emb.view(batch, N, 9)
        pos_emb = pos_emb.permute(0, 2, 1).contiguous()
        direct_vec = torch.tanh(self.dir_mlp(torch.cat([pos_emb, spatial_fea], dim=-2)))
        direct_vec = F.normalize(direct_vec, p=2, dim=1, eps=1e-08)
        direct = direct_vec.permute(0, 2, 1).contiguous()
        dir_emb = position_encoding(direct.view(-1, 3))
        dir_emb = dir_emb.view(batch, N, 9)
        dir_emb = dir_emb.permute(0, 2, 1).contiguous()
        distance = torch.abs(self.dist_mlp(torch.cat([pos_emb, dir_emb, spatial_fea], dim=-2)))
        deform_xyz = direct_vec * distance
        updated_pos = deform_xyz + parent_pos
        return (updated_pos, lsp_fea)

class PointNDF(nn.Module):

    def __init__(self, dim_feat=128):
        super(PointNDF, self).__init__()
        self.dim_feat = dim_feat
        self.encoder = PointTransformerV3(in_channels=3, cls_mode=True, enable_flash=False)
        self.ptv3_proj = nn.LazyLinear(dim_feat)
        self.denoise1 = DirectProj(dim=dim_feat)
        self.denoise2 = DirectProj(dim=dim_feat)
        self.denoise3 = DirectProj(dim=dim_feat)

    def forward(self, point_cloud):
        B, N, _ = point_cloud.shape
        device = point_cloud.device
        pcd = point_cloud.permute(0, 2, 1).contiguous()
        coord = point_cloud.reshape(-1, 3).contiguous()
        feat = coord
        offset = (torch.arange(B, device=device, dtype=torch.int32) + 1) * N
        data_dict = {'coord': coord, 'feat': feat, 'grid_size': 0.0066, 'offset': offset}
        pt = self.encoder(data_dict)
        x = pt.feat
        offset = pt.offset if hasattr(pt, 'offset') else pt['offset']
        pt_coord = pt.coord if hasattr(pt, 'coord') else pt['coord']
        starts = torch.cat([offset.new_zeros(1), offset[:-1]])
        pooled_list = []
        for s, e in zip(starts.tolist(), offset.tolist()):
            pooled_list.append(x[s:e].max(dim=0)[0])
        pooled = torch.stack(pooled_list, dim=0)
        if pooled.shape[1] != self.dim_feat:
            pooled = self.ptv3_proj(pooled)
        global_shape_fea = pooled.unsqueeze(-1)
        p1, f1 = self.denoise1(global_shape_fea, pcd, None)
        p2, f2 = self.denoise2(global_shape_fea, p1, f1)
        p3, f3 = self.denoise3(global_shape_fea, p2, f2)
        pcd1 = p1.permute(0, 2, 1).contiguous()
        pcd2 = p2.permute(0, 2, 1).contiguous()
        pcd3 = p3.permute(0, 2, 1).contiguous()
        return (pcd1, pcd2, pcd3)
