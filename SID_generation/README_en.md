# RQVAE Project

## Project Overview
This project implements the RQVAE (Residual Quantized Variational Autoencoder) model for feature encoding and decoding of images and videos. The model supports multi-layer residual quantization and is suitable for large-scale data training.

## Environment Dependencies
- Python 3.8+
- PyTorch 1.10+
- Other dependencies listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd open_RQVAE
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation
   ```bash
   wget -P datas/ https://mvap-public-data.oss-cn-zhangjiakou.aliyuncs.com/ICLR_2026_data/reconstruct_data_mask.npz
   wget -P datas/ https://mvap-public-data.oss-cn-zhangjiakou.aliyuncs.com/ICLR_2026_data/contrastive_data_mask.npz
   ```

## Training the Model
To start distributed training, use the following command:
```bash
python -m torch.distributed.launch --nnodes=2 --nproc_per_node=1 --master_port=27646 train.py --output_dir=/path/to/output --save_prefix=MODEL_NAME --cfg=configs/rqvae_i2v.yml
```

## Parameters
- `--cfg`: Path to the configuration file.
- `--output_dir`: Directory for model outputs.
- `--save_prefix`: Prefix for saving the model.

## Testing the Model

Use the following command to start testing:

```bash
python infer_SID.py
```

## Project Structure
```
open_RQVAE/
├── configs/            # Configuration files
├── data_loader/        # Data loading modules
├── rqvae_embed/        # Core RQVAE model code
├── utils/              # Utility functions
├── train.py            # Training script
├── infer_SID.py        # Inference script
└── requirements.txt    # Dependency list
```