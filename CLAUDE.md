# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CT2 (Colorization Transformer via Color Tokens) — ECCV 2022. Automatic colorization of grayscale images using Vision Transformers. Converts L-channel input through a ViT encoder + color-token decoder to predict AB chrominance in CIELAB space, quantized into 313 color classes.

## Commands

### Training
```bash
# Single GPU
python train.py --backbone vit_tiny_patch16_384 --decoder mask_transformer --dataset coco --batch-size 2
# Or via script
bash scripts/train.sh
```

### Evaluation
```bash
python -m segm.eval_metrics --pred_dir <colorized_dir> --gt_dir <groundtruth_dir>    # SSIM, PSNR, LPIPS
python -m segm.eval_fid --pred_dir <colorized_dir> --gt_dir <groundtruth_dir>        # FID
python -m segm.eval_colorfullness --pred_dir <colorized_dir>                          # Colorfulness
```

### Data Preparation
```bash
python -m segm.make_q_actual --dataset_dir /path/to/ImageNet --split val   # Generate color quantization ground truth
```

### Dependencies
```bash
pip install -r requirements.txt
```
Key pinned versions: `timm==0.4.12`, `mmcv==1.3.8`, `mmsegmentation==0.14.1`, `opencv-python==4.5.4.60`.

## Architecture

### Data Flow
1. ImageNet images loaded via `COCODataset` (`segm/data/coco.py`), converted RGB -> LAB
2. AB channels soft-encoded to 313 color classes via `SoftEncodeAB` (`segm/model/utils.py`)
3. L channel normalized and fed through ViT encoder -> decoder predicts color class distribution
4. `AnnealedMeanDecodeQ` converts predicted classes back to AB values (annealed softmax with temperature T=0.38)
5. Loss: `CrossEntropyLoss2d` on color classes + optional L1 on AB + optional VGG perceptual + optional edge (Sobel)
6. `RebalanceLoss` reweights gradients so rare colors contribute more

### Key Modules
- **`segm/model/segmenter.py:Segmenter`** — Main model. Wraps encoder+decoder, handles color encoding/decoding, loss rebalancing. Has separate `forward()` (training) and `inference()` paths.
- **`segm/model/factory.py`** — Model creation. `create_segmenter()` -> `create_vit()` + `create_decoder()`. Loads pretrained ViT weights from `.npz` files.
- **`segm/model/decoder.py`** — `DecoderLinear` (simple) and `MaskTransformer` (transformer-based with color token attention). MaskTransformer is the primary decoder.
- **`segm/engine.py`** — `train_one_epoch()` and `evaluate()`. Checkpoints every 5000 iterations.
- **`segm/data/factory.py`** — Dataset factory dispatches to `COCODataset` (ImageNet train/val) or `RandomDataset` (arbitrary image directory).

### Configuration
- **`configs/paths.yaml`** — All file paths (dataset, pretrained weights, mask prior, output dirs). Edit this before running.
- **`segm/config.yml`** — Model architectures (ViT variants, decoder configs) and dataset hyperparameters.
- **`segm/config.py`** — `load_paths()` loads `configs/paths.yaml`; `resolve_path()` resolves relative paths from project root; `load_config()` loads `segm/config.yml`.

### Color Space Details
- 313 color classes from AB gamut quantization (10-unit bins), defined by `ab-gamut.npy` and `q-prior.npy` in `segm/resources/`
- `mask_prior.pickle` (project root) maps L-channel ranges to likely color classes for optional mask-guided colorization
- `q_prior_all/L{1-4}-all.npy` — per-luminance color distribution priors

### Required Resources in `segm/resources/`
- ViT pretrained weights (`.npz` files from Google ViT)
- `ab-gamut.npy`, `q-prior.npy` — color gamut data
- VGG19 weights (for perceptual loss)
- Inception v3 weights (for FID evaluation)

## Codebase Conventions
- All scripts must be run from the project root directory
- Python modules use `segm.*` package imports (e.g., `from segm.config import load_paths`)
- Paths resolved via `segm.config.resolve_path()` relative to project root — never use `os.getcwd()` for path construction
- Dataset "coco" is actually ImageNet (historical naming from the original segmenter codebase)
- The `Loader` class (`segm/data/loader.py`) wraps PyTorch DataLoader with distributed sampler support
