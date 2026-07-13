# PePNet

Official core implementation of **PePNet: Pose-Enhanced Point Cloud Network for LiDAR-based Human Action Recognition in Outdoor Long-Range Scenarios**.

## Momo dataset

Download Momo separately from ModelScope:

https://modelscope.cn/datasets/shark3/momo

The training configuration expects two directories:

- `data.path`: Momo point-cloud pickle files.
- `data.pose_path`: matching HMR pose pickle files, using the same filenames as the point-cloud samples.

The dataset is not stored in this Git repository. After downloading and preparing it, set both paths in `config.yaml`.

## Repository structure

```text
.
├── config.yaml              # Core Momo/PePNet configuration
├── train.py                 # Training and evaluation entry point
├── datasets/momo.py         # Momo dataset loader
├── models/sequence_mamba.py # CPS2 and imPSTMamba
└── modules/                 # PePNet, Mamba, PSTConv, and CUDA operators
```

## Installation

PePNet requires a CUDA-capable PyTorch environment. Create an environment with a PyTorch build compatible with your CUDA toolkit, then install the Python dependencies:

```bash
pip install -r requirements.txt
```

Build the bundled PointNet++ CUDA operators:

```bash
cd modules
python setup.py install
cd ..
```

The CUDA toolkit used to compile the extension must be compatible with the installed PyTorch build.

## Configuration

The release keeps the paper's core hyperparameters in `config.yaml`. Before training, set:

```yaml
data:
  path: /path/to/momo/point_clouds
  pose_path: /path/to/momo/hmr_poses

training:
  output_dir: /path/to/outputs
```

The committed values of `data.path`, `data.pose_path`, and `training.output_dir` are intentionally empty. An empty output directory disables checkpoint and log writing. To resume training, set `training.resume` to a checkpoint path.

## Training

```bash
python train.py --config config.yaml
```

Select CUDA devices with `--gpu`, for example:

```bash
python train.py --config config.yaml --gpu 0
python train.py --config config.yaml --gpu 0,1
```

Training evaluates the test split after every epoch. When `training.output_dir` is configured, it writes `checkpoint_last.pth` and the best-performing `checkpoint_best.pth`.

## Citation

```bibtex
@article{liu2026pepnet,
  title   = {PePNet: Pose-Enhanced Point Cloud Network for LiDAR-based Human Action Recognition in Outdoor Long-Range Scenarios},
  author  = {Liu, Mengyuan and Deng, Zhichao and Zhang, Wanying and Wang, Ziyi and Li, Peiming and Liu, Jun},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026}
}
```
