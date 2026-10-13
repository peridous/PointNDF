from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
HERE = Path(__file__).resolve().parent
RESOLUTIONS = ('10000_poisson', '30000_poisson', '50000_poisson')

def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Train PointNDF on one GPU.')
    p.add_argument('--data-root', type=Path, default=HERE / 'data/punet')
    p.add_argument('--output', type=Path, required=True, help='New run directory (or same directory when resuming).')
    p.add_argument('--resume', type=Path, help='Full checkpoint written by this trainer.')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=5)
    p.add_argument('--device', type=int, default=0, help='Index within CUDA_VISIBLE_DEVICES.')
    p.add_argument('--val-resolution', choices=('10000_poisson', '50000_poisson'), default='10000_poisson')
    p.add_argument('--val-noise', type=float, choices=(0.01, 0.02, 0.03), default=0.02)
    p.add_argument('--patch-batch-size', type=int, default=8)
    p.add_argument('--val-seed', type=int, default=20260721)
    p.add_argument('--max-batches', type=int, default=0, help='Smoke tests only: cap training batches per epoch; zero means all.')
    p.add_argument('--val-limit', type=int, default=0, help='Smoke tests only: cap validation clouds; zero means all.')
    p.add_argument('--check-data', action='store_true', help='Inspect file layout and counts without importing CUDA dependencies.')
    a = p.parse_args(argv)
    for key in ('epochs', 'batch_size', 'patch_batch_size'):
        if getattr(a, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    for key in ('workers', 'device', 'max_batches', 'val_limit', 'seed', 'val_seed'):
        if getattr(a, key) < 0:
            p.error(f"--{key.replace('_', '-')} must be nonnegative")
    a.data_root = a.data_root.resolve()
    a.output = a.output.resolve()
    if a.resume:
        a.resume = a.resume.resolve()
    return a

def inspect_data(args):
    inventory = {}
    for resolution in RESOLUTIONS:
        path = args.data_root / 'pointclouds/train' / resolution
        names = sorted((p.name for p in path.glob('*.xyz')))
        if not names:
            raise ValueError(f'No training XYZ files: {path}')
        inventory[resolution] = names
    clean = args.data_root / 'pointclouds/test/clean' / args.val_resolution
    noisy = args.data_root / 'pointclouds/test/noise' / f'{args.val_resolution}_{args.val_noise}'
    names = sorted((p.name for p in clean.glob('*.xyz')))
    if not names or set(names) != {p.name for p in noisy.glob('*.xyz')}:
        raise ValueError(f'Clean/noisy validation files must match and be nonempty: {clean}, {noisy}')
    inventory['validation'] = names
    if sum((len(inventory[r]) for r in RESOLUTIONS)) * 100 < args.batch_size:
        raise ValueError('Batch size exceeds the number of training patches.')
    return inventory

def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seed_worker(_worker_id):
    import torch
    import numpy as np
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)

def capture_rng(generator):
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), loader=generator.get_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(state, generator):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])

def make_scheduler(optimizer):
    from torch.optim.lr_scheduler import StepLR
    from pointndf.schedule import GradualWarmupScheduler
    return GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=20, after_scheduler=StepLR(optimizer, step_size=20, gamma=0.5))

def scheduler_state(scheduler):
    state = dict(scheduler.state_dict())
    state.pop('after_scheduler', None)
    return dict(warmup=state, step=scheduler.after_scheduler.state_dict())

def load_scheduler_state(scheduler, state):
    scheduler.load_state_dict(state['warmup'])
    scheduler.after_scheduler.load_state_dict(state['step'])

def atomic_save(state, path):
    import torch
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temp)
    os.replace(temp, path)

def validate(model, dataset, args, device):
    import numpy as np
    import torch
    from pointndf.layers import fps_subsample
    from pointndf.data import split_patches
    from pointndf.losses import chamfer_distance_unit_sphere
    from pointndf.transforms import NormalizeUnitSphere
    model.eval()
    values = []
    count = len(dataset) if not args.val_limit else min(args.val_limit, len(dataset))
    with torch.no_grad():
        for index in range(count):
            seed_all(args.val_seed + index)
            data = dataset[index]
            noisy, center, scale = NormalizeUnitSphere.normalize(data['pcl_noisy'])
            noisy = torch.as_tensor(noisy, dtype=torch.float32, device=device)
            center = torch.as_tensor(center, dtype=torch.float32, device=device)
            scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
            gt = ((data['pcl_clean'].to(device) - center) / scale).unsqueeze(0)
            patches = split_patches(noisy.cpu(), n_patches=30 if args.val_resolution == '10000_poisson' else 120, nsample=1024)
            outputs = []
            for start in range(0, len(patches), args.patch_batch_size):
                outputs.append(model(patches[start:start + args.patch_batch_size].to(device).contiguous())[2])
            candidates = torch.cat(outputs).reshape(1, -1, 3)
            prediction = fps_subsample(candidates, gt.shape[1])
            cd = float(chamfer_distance_unit_sphere(prediction, gt).item() * 10000.0)
            if not np.isfinite(cd):
                raise RuntimeError(f"Nonfinite validation CD: {data['name']}")
            values.append(cd)
            print(f"validation {index + 1}/{count}: {data['name']} CD={cd:.6f}", flush=True)
    return float(np.mean(values))

def main(argv=None):
    args = parse_args(argv)
    inventory = inspect_data(args)
    print(json.dumps({k: len(v) for k, v in inventory.items()}, indent=2))
    if args.check_data:
        return
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This entrypoint supports one GPU. Run directly, without multi-process torchrun.')
    if args.output.exists() and any(args.output.iterdir()):
        if not args.resume or args.resume.parent != args.output:
            raise ValueError('Nonempty output requires resuming a checkpoint from that same directory.')
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the model and point-cloud extensions.')
    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    from pointndf.model import PointNDF
    from pointndf.data import PUNetDataset, PUNetTestDataset, PairedPatchDataset
    from pointndf.transforms import standard_train_transforms
    from pointndf.losses import get_loss
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(args.seed)
    datasets = [PUNetDataset(root=str(args.data_root), split='train', resolution=r, transform=standard_train_transforms(noise_std_min=0.005, noise_std_max=0.03, rotate=True, scale_d=0.2)) for r in RESOLUTIONS]
    for dataset in datasets:
        pairs = sorted(zip(dataset.pointcloud_names, dataset.pointclouds))
        dataset.pointcloud_names, dataset.pointclouds = map(list, zip(*pairs))
    train_set = PairedPatchDataset(datasets=datasets, patch_size=1024, num_patches=100, transform=None)
    val_set = PUNetTestDataset(root=str(args.data_root), resolution=args.val_resolution, level=args.val_noise)
    triples = sorted(zip(val_set.pointcloud_names, val_set.clean_pointclouds, val_set.noise_pointclouds))
    val_set.pointcloud_names, val_set.clean_pointclouds, val_set.noise_pointclouds = map(list, zip(*triples))
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, drop_last=True, shuffle=False, worker_init_fn=seed_worker, generator=generator, persistent_workers=False)
    model = PointNDF().to(device)
    model.ptv3_proj.initialize_parameters(torch.empty(1, 512, device=device))
    optimizer = torch.optim.Adam([{'params': model.encoder.parameters(), 'lr': 5e-05}, {'params': [p for n, p in model.named_parameters() if not n.startswith('encoder.')], 'lr': 0.0005}], betas=(0.8, 0.9), weight_decay=0)
    scheduler = make_scheduler(optimizer)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['protocol'] = 'reconstructed-single-gpu-v1'
    start_epoch, steps, best = (0, 0, float('inf'))
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        required = {'format', 'epoch_index', 'steps', 'best_metrics', 'model', 'optimizer', 'scheduler', 'rng', 'config', 'inventory'}
        if not required <= saved.keys() or saved['format'] != 'pointndf-training-v1':
            raise ValueError('Resume requires a full checkpoint from this entrypoint, not the historical inference checkpoint.')
        for key in config.keys() - {'epochs', 'output', 'resume', 'check_data', 'device'}:
            if config[key] != saved['config'].get(key):
                raise ValueError(f'Resume configuration changed: {key}')
        if inventory != saved['inventory']:
            raise ValueError('Dataset file inventory changed since checkpoint.')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        load_scheduler_state(scheduler, saved['scheduler'])
        start_epoch, steps, best = (saved['epoch_index'], saved['steps'], saved['best_metrics'])
        restore_rng(saved['rng'], generator)
    if start_epoch >= args.epochs:
        raise ValueError('--epochs must exceed the resumed epoch.')
    args.output.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), HERE / 'pointndf/model.py', HERE / 'pointndf/backbone/model.py', HERE / 'pointndf/layers.py', HERE / 'pointndf/data.py', HERE / 'pointndf/transforms.py', HERE / 'pointndf/losses.py', HERE / 'pointndf/schedule.py']
    hashes = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    if args.resume and saved.get('source_hashes') != hashes:
        raise ValueError('Training source changed since checkpoint; resume with the original source.')
    provenance = dict(config=config, inventory=inventory, source_hashes=hashes, torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device), python=sys.version, parameters=sum((p.numel() for p in model.parameters())), batches_per_epoch=len(loader))
    (args.output / f'run_from_epoch_{start_epoch:03d}.json').write_text(json.dumps(provenance, indent=2) + '\n')
    snapshot = args.output / 'source'
    for p in source_files:
        target = snapshot / p.relative_to(HERE)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(p.read_bytes())
    print(f"{provenance['parameters']:,} parameters; {len(loader)} batches/epoch", flush=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        losses, batches = (0.0, 0)
        for data in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(data['pcl_noisy'].to(device, non_blocking=True))
            loss, _ = get_loss(outputs, data['pcl_clean'].to(device, non_blocking=True), use_mse=False, sqrt=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss at epoch {epoch}, batch {batches + 1}')
            loss.backward()
            optimizer.step()
            if steps <= 20:
                scheduler.step()
            steps += 1
            losses += loss.item()
            batches += 1
            if batches % 25 == 0:
                print(f'epoch {epoch} batch {batches}/{len(loader)} loss={loss.item():.6f}', flush=True)
            if args.max_batches and batches >= args.max_batches:
                break
        scheduler.step()
        rng = capture_rng(generator)
        try:
            metric = validate(model, val_set, args, device)
        finally:
            restore_rng(rng, generator)
        improved = metric < best
        best = min(best, metric)
        state = dict(format='pointndf-training-v1', epoch_index=epoch, steps=steps, best_metrics=best, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler_state(scheduler), rng=capture_rng(generator), config=config, inventory=inventory, source_hashes=hashes)
        atomic_save(state, args.output / 'last.pth')
        if improved:
            atomic_save(state, args.output / 'best.pth')
        record = dict(epoch=epoch, train_loss_scaled=losses / batches, val_cd_x1e4=metric, best_cd_x1e4=best, next_lr=[g['lr'] for g in optimizer.param_groups], steps=steps)
        with (args.output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
if __name__ == '__main__':
    main()
