import torch
from torch import nn, einsum
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, ball_query, three_nn, three_interpolate, grouping_operation

class Conv1d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, if_bn=True, activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)
        if self.activation_fn is not None:
            out = self.activation_fn(out)
        return out

class Conv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True, activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)
        if self.activation_fn is not None:
            out = self.activation_fn(out)
        return out

class MLP(nn.Module):

    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Linear(last_channel, out_channel))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Linear(last_channel, layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_CONV(nn.Module):

    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_Res(nn.Module):

    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out

def MLP_Stacks(channels, bn=None):
    layers = []
    last_channel = channels[0]
    for out_channel in channels[1:-1]:
        layers.append(MLP_Res(last_channel, None, out_channel))
        if bn:
            layers.append(nn.BatchNorm1d(out_channel))
        layers.append(nn.ReLU())
        last_channel = out_channel
    layers.append(MLP_Res(last_channel, None, channels[-1]))
    return nn.Sequential(*layers)

def sample_and_group(xyz, points, npoint, nsample, radius, use_xyz=True):
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint))
    idx = ball_query(radius, nsample, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, nsample)
    if points is not None:
        grouped_points = grouping_operation(points, idx)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return (new_xyz, new_points, idx, grouped_xyz)

def sample_and_group_all(xyz, points, use_xyz=True):
    b, _, nsample = xyz.shape
    device = xyz.device
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float, device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample, device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz
    return (new_xyz, new_points, idx, grouped_xyz)

class PointNet_SA_Module(nn.Module):

    def __init__(self, npoint, nsample, radius, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True):
        super(PointNet_SA_Module, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        if use_xyz:
            in_channel += 3
        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points):
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(xyz, points, self.npoint, self.nsample, self.radius, self.use_xyz)
        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]
        return (new_xyz, new_points)

class PointNet_FP_Module(nn.Module):

    def __init__(self, in_channel, mlp, use_points1=False, in_channel_points1=None, if_bn=True):
        super(PointNet_FP_Module, self).__init__()
        self.use_points1 = use_points1
        if use_points1:
            in_channel += in_channel_points1
        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz1, xyz2, points1, points2):
        dist, idx = three_nn(xyz1.permute(0, 2, 1).contiguous(), xyz2.permute(0, 2, 1).contiguous())
        dist = torch.clamp_min(dist, 1e-10)
        recip_dist = 1.0 / dist
        norm = torch.sum(recip_dist, 2, keepdim=True).repeat((1, 1, 3))
        weight = recip_dist / norm
        interpolated_points = three_interpolate(points2, idx, weight)
        if self.use_points1:
            new_points = torch.cat([interpolated_points, points1], 1)
        else:
            new_points = interpolated_points
        new_points = self.mlp_conv(new_points)
        return new_points

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_knn(nsample, xyz, new_xyz, include_self=True):
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)
    sorted_dist, indices = torch.sort(sqrdists, dim=-1, descending=False)
    idx = indices[:, :, pad:nsample + pad]
    sdist = sorted_dist[:, :, pad:nsample + pad]
    return (idx.int(), sdist)

def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint))
    if idx is None:
        idx, sqdists = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)
    if points is not None:
        grouped_points = grouping_operation(points, idx)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return (new_xyz, new_points, idx, grouped_xyz, sqdists)

class PointNet_SA_Module_KNN(nn.Module):

    def __init__(self, npoint, nsample, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True, if_idx=False):
        super(PointNet_SA_Module_KNN, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3
        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points, idx=None):
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz, sqdists = sample_and_group_knn(xyz, points, self.npoint, self.nsample, self.use_xyz, idx=idx)
        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]
        if self.if_idx:
            return (new_xyz, new_points, idx)
        else:
            return (new_xyz, new_points)

def fps_subsample(pcd, n_points=2048):
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd

class Transformer(nn.Module):

    def __init__(self, in_channel, dim=256, n_knn=32, pos_hidden_dim=64, attn_hidden_multiplier=2):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)
        self.pos_mlp = nn.Sequential(nn.Conv2d(3, pos_hidden_dim, 1), nn.BatchNorm2d(pos_hidden_dim), nn.ReLU(), nn.Conv2d(pos_hidden_dim, dim, 1))
        self.attn_mlp = nn.Sequential(nn.Conv2d(dim, dim * attn_hidden_multiplier, 1), nn.BatchNorm2d(dim * attn_hidden_multiplier), nn.ReLU(), nn.Conv2d(dim * attn_hidden_multiplier, dim, 1))
        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        identity = x
        x = self.linear_start(x)
        b, dim, n = x.shape
        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn, sqdist = query_knn(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)
        key = grouping_operation(key, idx_knn)
        qk_rel = query.reshape((b, -1, n, 1)) - key
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)
        pos_embedding = self.pos_mlp(pos_rel)
        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)
        value = grouping_operation(value, idx_knn) + pos_embedding
        agg = einsum('b c i j, b c i j -> b c i', attention, value)
        y = self.linear_end(agg)
        return y + identity

class ShapeAttention(nn.Module):

    def __init__(self, in_channel, dim=64, n_knn=32, pos_hidden_dim=64, attn_hidden_multiplier=2):
        super(ShapeAttention, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1, bias=False)
        self.conv_value = nn.Conv1d(dim, dim, 1, bias=False)
        self.pos_mlp = nn.Sequential(nn.Conv2d(3, pos_hidden_dim, 1), nn.BatchNorm2d(pos_hidden_dim), nn.ReLU(), nn.Conv2d(pos_hidden_dim, dim, 1))
        self.shape_query = nn.Sequential(nn.Conv2d(10, pos_hidden_dim, 1), nn.BatchNorm2d(pos_hidden_dim), nn.ReLU(), nn.Conv2d(pos_hidden_dim, dim, 1, bias=False))
        self.attn_mlp = nn.Sequential(nn.Conv2d(dim, dim * attn_hidden_multiplier, 1), nn.BatchNorm2d(dim * attn_hidden_multiplier), nn.ReLU(), nn.Conv2d(dim * attn_hidden_multiplier, dim, 1))
        self.linear_start = MLP_CONV(in_channel=in_channel * 2, layer_dims=[128, dim], bn=True)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        x = self.linear_start(torch.cat([x, torch.max(x, 2, keepdim=True)[0].repeat((1, 1, x.size(2)))], 1))
        b, c, n = x.shape
        identity = x
        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn, sqdist = query_knn(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        coord_xi = pos.view(b, -1, n, 1).repeat(1, 1, 1, self.n_knn)
        coord_xj = grouping_operation(pos, idx_knn)
        pos_rel = coord_xi - coord_xj
        dis_xi_xj = torch.norm(pos_rel, p=2, dim=1).unsqueeze(1)
        h_xi_xj = torch.cat((dis_xi_xj, coord_xi, coord_xj, pos_rel), dim=1)
        query = self.shape_query(h_xi_xj)
        key = grouping_operation(key, idx_knn)
        qk_rel = query * key
        pos_embedding = self.pos_mlp(pos_rel)
        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)
        value = grouping_operation(value, idx_knn) + pos_embedding
        agg = einsum('b c i j, b c i j -> b c i', attention, value)
        out = self.linear_end(agg)
        return out
