import argparse
import os
from pathlib import Path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def strip_module_prefix(state):
    return {key[7:] if key.startswith('module.') else key: value for key, value in state.items()}

def as_tensor(value, device):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=torch.float32)

def save_xyz(path, points):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, points, fmt='%.8f')

def load_model(device):
    print('Loading checkpoint:', CHECKPOINT)
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    state = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state = strip_module_prefix(state)
    model = Model()
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f'Checkpoint mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}')
    model = model.to(device)
    model.eval()
    print('Checkpoint loaded with strict=True.')
    return model

def evaluate_setting(model, resolution, noise, patch_batch_size, device):
    dataset = PUNetTestDataset(root=DATA_ROOT, resolution=resolution, level=noise)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True, shuffle=False)
    n_patches = 30 if resolution == '10000_poisson' else 120
    stage_totals = np.zeros(3, dtype=np.float64)
    sample_count = 0
    setting_dir = os.path.join(OUTPUT_ROOT, f'{resolution}_{noise:.2f}')
    progress = tqdm(loader, desc=f'{resolution}, noise={noise:.2f}')
    for sample_index, data in enumerate(progress):
        set_seed(SEED + sample_index)
        noisy_world = data['pcl_noisy'].squeeze(0)
        clean_world = data['pcl_clean'].squeeze(0)
        name = data['name'][0]
        noisy_unit, center, scale = NormalizeUnitSphere.normalize(noisy_world.squeeze())
        noisy_unit = as_tensor(noisy_unit, device)
        center = as_tensor(center, device)
        scale = as_tensor(scale, device)
        clean_world = as_tensor(clean_world, device)
        clean_unit = (clean_world - center) / scale
        gt_unit = clean_unit.unsqueeze(0)
        grouped_patches = split_patches(noisy_unit.detach().cpu(), n_patches=n_patches, nsample=1024)
        stage_patch_outputs = [[], [], []]
        for start in range(0, grouped_patches.shape[0], patch_batch_size):
            patch_batch = grouped_patches[start:start + patch_batch_size].to(device, non_blocking=True)
            outputs = model(patch_batch.contiguous())
            if len(outputs) != 3:
                raise RuntimeError(f'Expected three model outputs, received {len(outputs)}')
            for stage_index in range(3):
                stage_patch_outputs[stage_index].append(outputs[stage_index].detach())
        n_points = clean_world.shape[0]
        sampled_stages = []
        for stage_index in range(3):
            candidates = torch.cat(stage_patch_outputs[stage_index], dim=0).reshape(1, -1, 3)
            if candidates.shape[1] < n_points:
                raise RuntimeError(f'Stage {stage_index + 1}: only {candidates.shape[1]} candidates for {n_points} outputs')
            sampled = fps_subsample(candidates, n_points)
            sampled_stages.append(sampled)
        cds = []
        for stage_index, prediction in enumerate(sampled_stages):
            cd = chamfer_distance_unit_sphere(prediction, gt_unit).item() * 10000.0
            stage_totals[stage_index] += cd
            cds.append(cd)
            prediction_world = prediction * scale + center
            output_path = os.path.join(setting_dir, f'stage{stage_index + 1}', f'{name}.xyz')
            save_xyz(output_path, prediction_world.squeeze(0).cpu().numpy())
        sample_count += 1
        progress.set_postfix(s1=f'{cds[0]:.4f}', s2=f'{cds[1]:.4f}', s3=f'{cds[2]:.4f}')
    averages = stage_totals / sample_count
    print()
    print('=' * 68)
    print('Resolution:', resolution)
    print('Noise:', noise)
    print('Number of shapes:', sample_count)
    print('Stage 1 CD x1e4: %.4f' % averages[0])
    print('Stage 2 CD x1e4: %.4f' % averages[1])
    print('Stage 3 CD x1e4: %.4f' % averages[2])
    print('Saved outputs to:', setting_dir)
    print('=' * 68)
    return averages

def main():
    parser = argparse.ArgumentParser(description='Denoise the fixed PU-Net or ModelNet10 evaluation sets.')
    parser.add_argument('--dataset', choices=('punet', 'modelnet10'), required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=('pointndf', 'one_stage', 'two_stage', 'direct_displacement', 'unit_direction', 'pointnetpp'), default='pointndf')
    parser.add_argument('--resolution', choices=('10000_poisson', '50000_poisson'))
    parser.add_argument('--noise', type=float, choices=(0.01, 0.02, 0.03))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--patch-batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260721)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if not args.all and (args.resolution is None or args.noise is None):
        parser.error('Specify --all, or both --resolution and --noise.')
    if args.patch_batch_size < 1:
        parser.error('--patch-batch-size must be positive.')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose an empty output directory.')
    global random, np, torch, tqdm, Model, fps_subsample, PUNetTestDataset, split_patches
    global chamfer_distance_unit_sphere, NormalizeUnitSphere, CHECKPOINT, DATA_ROOT, OUTPUT_ROOT, SEED
    import random
    import importlib
    import numpy as np
    import torch
    from tqdm import tqdm
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for model inference.')
    torch.cuda.set_device(args.device)
    from pointndf.layers import fps_subsample
    from pointndf.data import PUNetTestDataset, split_patches
    from pointndf.losses import chamfer_distance_unit_sphere
    from pointndf.transforms import NormalizeUnitSphere
    module = 'pointndf.model' if args.model == 'pointndf' else 'pointndf.ablations.' + args.model
    Model = importlib.import_module(module).PointNDF
    CHECKPOINT = str(args.checkpoint.resolve())
    DATA_ROOT = str((args.data_root or Path(__file__).resolve().parent / 'data' / args.dataset).resolve())
    OUTPUT_ROOT = str(args.output.resolve())
    SEED = args.seed
    set_seed(SEED)
    device = torch.device('cuda', args.device)
    model = load_model(device)
    settings = [(r, n) for r in ('10000_poisson', '50000_poisson') for n in (0.01, 0.02, 0.03)] if args.all else [(args.resolution, args.noise)]
    import json
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'run.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2) + '\n')
    summary = {}
    with torch.no_grad():
        for resolution, noise in settings:
            result = evaluate_setting(model, resolution, noise, args.patch_batch_size, device)
            summary[f'{resolution}_{noise:.2f}'] = result.tolist()
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
if __name__ == '__main__':
    main()
