import torch
from pointndf.ops.chamfer import chamfer_3DDist
import pytorch3d.loss
chamfer_dist = chamfer_3DDist()

def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)

def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2

def normalize_sphere(pc, radius=1.0):
    p_max = pc.max(dim=-2, keepdim=True)[0]
    p_min = pc.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2
    pc = pc - center
    scale = (pc ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / radius
    pc = pc / scale
    return (pc, center, scale)

def normalize_std(pc, std=1.0):
    center = pc.mean(dim=-2, keepdim=True)
    pc = pc - center
    scale = pc.view(pc.size(0), -1).std(dim=-1).view(pc.size(0), 1, 1) / std
    pc = pc / scale
    return (pc, center, scale)

def normalize_pcl(pc, center, scale):
    return (pc - center) / scale

def denormalize_pcl(pc, center, scale):
    return pc * scale + center

def chamfer_distance_unit_sphere(gen, ref, batch_reduction='mean', point_reduction='mean'):
    ref, center, scale = normalize_sphere(ref)
    gen = normalize_pcl(gen, center, scale)
    cd, norm = pytorch3d.loss.chamfer_distance(gen, ref, batch_reduction=batch_reduction, point_reduction=point_reduction)
    return cd

def get_loss(pcds_pred, gt, use_mse=False, sqrt=True):
    if sqrt:
        CD = chamfer_sqrt
    else:
        CD = chamfer_distance_unit_sphere
    MSE = torch.nn.MSELoss()
    pcd0, pcd1, pcd2 = pcds_pred
    if use_mse:
        cd1 = torch.sqrt(MSE(pcd0, gt))
        cd2 = torch.sqrt(MSE(pcd1, gt))
        cd3 = torch.sqrt(MSE(pcd2, gt))
    else:
        cd1 = CD(pcd0, gt)
        cd2 = CD(pcd1, gt)
        cd3 = CD(pcd2, gt)
    loss_sum = (1 * cd1 + 1 * cd2 + 1 * cd3) * 1000.0
    losses = [cd1, cd2, cd3]
    return (loss_sum, losses)
