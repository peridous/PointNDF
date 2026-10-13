# PointNDF

This is the code for our paper, *PointNDF: Cascaded Neural Displacement Fields for Point Cloud Denoising*.

## Setup

We used Python 3.10, PyTorch 2.5.1, and CUDA 11.8.

```bash
python -m pip install -r requirements.txt
python -m pip install -e third_party/pointnet2
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@b6a77ad7aaf41ed90fca80ce6a2bac3c462a7881"
```

The Chamfer extension in `pointndf/ops` is compiled the first time it is imported.

## Data

Download PU-Net and ModelNet10 separately. The PU-Net folders should look like this:

```text
PUNet/
  pointclouds/
    train/{10000,30000,50000}_poisson/
    test/clean/{10000,50000}_poisson/
    test/noise/{10000,50000}_poisson_{0.01,0.02,0.03}/
  meshes/test/
```

ModelNet10 uses the same test folder layout. Put its meshes in `meshes/test/10000_poisson/`.

## Training

Check that the dataset is in the right place:

```bash
python train.py --data-root /path/to/PUNet --output runs/check --check-data
```

Before a full run, try one batch on the GPU:

```bash
python train.py --data-root /path/to/PUNet --output runs/smoke --epochs 1 --batch-size 2 --workers 0 --max-batches 1 --val-limit 1 --patch-batch-size 2
```

Train the model:

```bash
python train.py --data-root /path/to/PUNet --output runs/pointndf
```

## Inference

```bash
python infer.py --dataset punet --data-root /path/to/PUNet --checkpoint /path/to/pointndf.pth --resolution 10000_poisson --noise 0.01 --output runs/punet_10k_001
```

Use `--all` to run every resolution and noise level.

## Evaluation

```bash
python evaluate.py --dataset punet --data-root /path/to/data --predictions /path/to/predictions --resolution 10000 --noise 0.01 --output runs/metrics
```

Add `--p2m` to compute point-to-mesh distance.

The original checkpoint loads correctly and the model has 39,119,148 parameters. The new training entrypoint has not been tested on the GPU yet.

Third-party notices are listed in `THIRD_PARTY.md` and `licenses/`.
