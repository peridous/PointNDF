import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent


def load_points(path):
    points = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if points.shape[1] < 3 or len(points) == 0 or not np.isfinite(points[:, :3]).all():
        raise ValueError(f'Invalid XYZ file: {path}')
    return points[:, :3]


def squared_chamfer(prediction, reference):
    center = (reference.max(0) + reference.min(0)) / 2
    scale = np.linalg.norm(reference - center, axis=1).max()
    if scale <= 0:
        raise ValueError('Reference cloud has zero radius.')
    prediction = (prediction - center) / scale
    reference = (reference - center) / scale
    return float(np.square(cKDTree(reference).query(prediction)[0]).mean() + np.square(cKDTree(prediction).query(reference)[0]).mean())


def mesh_distance(prediction, path, device):
    import open3d as o3d
    import torch
    from pytorch3d.loss import point_mesh_face_distance
    from pytorch3d.structures import Meshes, Pointclouds
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
    faces = torch.as_tensor(np.asarray(mesh.triangles), dtype=torch.int64, device=device)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f'Invalid mesh: {path}')
    center = (vertices.max(0).values + vertices.min(0).values) / 2
    scale = (vertices - center).norm(dim=-1).max()
    if scale <= 0:
        raise ValueError(f'Mesh has zero radius: {path}')
    points = (torch.as_tensor(prediction, dtype=torch.float32, device=device) - center) / scale
    return float(point_mesh_face_distance(Meshes([(vertices-center)/scale], [faces]), Pointclouds([points])).item())


def main():
    p = argparse.ArgumentParser(description='Score saved predictions using reference-normalized squared CD and optional bidirectional squared P2M.')
    p.add_argument('--manifest', type=Path)
    p.add_argument('--dataset', choices=('punet', 'modelnet10'))
    p.add_argument('--method')
    p.add_argument('--predictions', type=Path)
    p.add_argument('--resolution', type=int, choices=(10000, 50000))
    p.add_argument('--noise', type=float, choices=(.01, .02, .03))
    p.add_argument('--data-root', type=Path, default=ROOT/'data')
    p.add_argument('--p2m', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Choose an empty output directory.')
    if args.predictions:
        if args.dataset is None or args.resolution is None or args.noise is None:
            p.error('--predictions requires --dataset, --resolution and --noise.')
        settings = [dict(dataset=args.dataset, method=args.method or 'pointndf', resolution=args.resolution, noise=args.noise, directory=str(args.predictions.resolve()))]
    elif args.manifest:
        settings = json.loads(args.manifest.read_text())
        for key in ('dataset', 'method', 'resolution', 'noise'):
            value = getattr(args, key)
            if value is not None:
                settings = [s for s in settings if s[key] == value]
    else:
        p.error('Provide --predictions or --manifest.')
    if not settings:
        p.error('No matching settings.')
    summaries, per_shape = [], []
    for setting in settings:
        dataset = setting['dataset']
        base = args.data_root/dataset
        prediction_dir = Path(setting['directory'])
        if not prediction_dir.is_absolute():
            prediction_dir = ROOT/prediction_dir
        clean_dir = base/'pointclouds/test/clean'/f'{setting["resolution"]}_poisson'
        predictions = {path.stem.lower():path for path in prediction_dir.glob('*.xyz')}
        references = {path.stem.lower():path for path in clean_dir.glob('*.xyz')}
        if len(references) != 20 or set(predictions) != set(references):
            raise ValueError(f'Expected 20 matching predictions and references: {prediction_dir}')
        meshes = {}
        if args.p2m:
            mesh_dir = base/'meshes/test'
            if dataset == 'modelnet10':
                mesh_dir = mesh_dir/'10000_poisson'
            meshes = {path.stem.lower():path for path in mesh_dir.glob('*.off')}
            if not set(references) <= set(meshes):
                raise ValueError(f'Missing meshes: {mesh_dir}')
        rows = []
        for name in sorted(references):
            pred = load_points(predictions[name])
            ref = load_points(references[name])
            row = dict(dataset=dataset, method=setting['method'], resolution=setting['resolution'], noise=setting['noise'], shape=name, points=len(pred), cd_x1e4=squared_chamfer(pred,ref)*1e4)
            if args.p2m:
                row['p2m_x1e4'] = mesh_distance(pred, meshes[name], args.device)*1e4
            rows.append(row)
        summary = {k: rows[0][k] for k in ('dataset','method','resolution','noise')}
        summary['shapes'] = len(rows)
        for metric in ('cd_x1e4','p2m_x1e4') if args.p2m else ('cd_x1e4',):
            summary[metric] = float(np.mean([row[metric] for row in rows]))
        summaries.append(summary)
        per_shape.extend(rows)
        print(json.dumps(summary), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('summary.csv',summaries), ('per_shape.csv',per_shape)):
        with (args.output/name).open('w') as f:
            writer = csv.DictWriter(f,fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
