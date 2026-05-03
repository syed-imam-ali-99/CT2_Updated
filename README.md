# CT<sup>2</sup>: Colorization Transformer via Color Tokens

## Introduction
This is the official PyTorch **CT<sup>2</sup>** implementation.

We present **C**olorization **T**ransformer via **C**olor **T**okens (**CT<sup>2</sup>**) to colorize grayish images while dealing with incorrect semantic colors and undersaturation without any additional external priors.

<img src="https://github.com/shuchenweng/CT2/blob/main/application.png" align=center />

## Prerequisites
* Python 3.10+
* PyTorch 1.10+
* NVIDIA GPU + CUDA cuDNN

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Download the pretrained ViT-Large model and place it in `segm/resources/`:
```
https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz
```

## Configuration

All file paths are centralized in `configs/paths.yaml`. Edit this file before running any scripts:

```yaml
dataset_dir: "/path/to/ImageNet"       # Dataset location
output_dir: "results/imagenet"         # Where colorized images are saved
log_dir: "logs"                        # Training logs and checkpoints
```

Pretrained weights, resource paths, and evaluation directories are also configured here.

## Usage

All scripts are in `scripts/`. Run from the project root with your venv activated:
```bash
source .venv/bin/activate
```

### 1) Training (Multi-GPU)
```bash
bash scripts/train.sh
```
Key parameters in `scripts/train.sh`:
- `NUM_GPUS=8` — number of GPUs
- `BATCH_SIZE=48` — total batch size (divided across GPUs)
- `BACKBONE="vit_large_patch16_384"` — model architecture
- `LOG_DIR="segm/vit-large"` — checkpoint directory

### 2) Inference (Colorize Images)
```bash
bash scripts/inference.sh
```
Key parameters in `scripts/inference.sh`:
- `LOG_DIR="segm/vit-large"` — directory containing `checkpoint.pth`
- `OUTPUT_DIR="results/coco"` — where colorized images are saved
- `DATASET="coco"` — use `"random"` for custom image folders
- `DATASET_DIR=""` — leave empty to use `configs/paths.yaml`, or set to a folder path

Download pretrained weights: [Baidu Pan](https://pan.baidu.com/s/1cak_aAHIaMTVpTLP0yqRyw) (code: *v4ay*) or [Google Drive](https://drive.google.com/file/d/15LsqvHu1_g6OEEUbir7i1wbLDxIP4LTU/view?usp=sharing). Rename to `checkpoint.pth` and place in your `LOG_DIR`.

### 3) Evaluation
```bash
bash scripts/eval_metrics.sh        # SSIM, PSNR, LPIPS
bash scripts/eval_fid.sh            # FID score
bash scripts/eval_colorfulness.sh   # Colorfulness metric
```
Pass `--pred_dir` and `--gt_dir` as arguments, or set `PRED_DIR`/`GT_DIR` environment variables.

### 4) Data Preparation
```bash
bash scripts/make_q_actual.sh       # Generate color quantization ground truth
```

## Project Structure
```
configs/paths.yaml       # Centralized path configuration
scripts/                 # Bash scripts for all operations
train.py                 # Training entry point (multi-GPU, DDP)
test.py                  # Inference entry point (single-GPU)
segm/
  config.py              # Config loader (paths.yaml + config.yml)
  config.yml             # Model architectures & hyperparameters
  engine.py              # Training/evaluation loops
  metrics.py             # SSIM, PSNR, LPIPS, FID computation
  model/                 # ViT encoder, decoder, segmenter
  data/                  # Dataset loaders (ImageNet, custom)
  optim/                 # Optimizer and LR scheduler
  utils/                 # Distributed training, logging
  resources/             # Pretrained weights (ViT, VGG19, etc.)
```

## License
Licensed under [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).
